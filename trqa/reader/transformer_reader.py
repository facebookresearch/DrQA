from transformers.data.processors.squad import squad_convert_examples_to_features, SquadExample, SquadResult
from transformers import (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,
                          BertConfig, BertForQuestionAnswering, BertTokenizer,
                          XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
                          XLMConfig, XLMForQuestionAnswering, XLMTokenizer,
                          DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer,
                          AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer,)
from multiprocessing import cpu_count
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import os
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
import json

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}

def to_list(tensor):
    return tensor.detach().cpu().tolist()


class TransformerReader(object):
    def __init__(self, model_type='roberta',
                 model_name_or_path='/net/csefiles/siemens/yzhang952/erenup/transformers-old/models_roberta/large_qa_ranked_negtive_sp_para_4',
                 lang_id=0,
                 no_cuda=False,
                 per_gpu_eval_batch_size=2,
                 n_best_size=20,
                 max_answer_length=30,
                 ):

        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

        self.lang_id = lang_id
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(
            model_name_or_path,
            do_lower_case=True,
        )
        self.model = model_class.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        self.model.to(self.device)
        logger.info('qa model from: {}'.format(model_name_or_path))

    def answer_question(self, ranked_examples):
        squad_examples = [SquadExample(
            qas_id=str(x['id']),
            question_text=x['question'],
            context_text=x['document'],
            answer_text=None,
            start_position_character=None,
            title='',
            answers=[],
        ) for x in ranked_examples]

        squad_features, squad_dataset = squad_convert_examples_to_features(
            examples=squad_examples,
            tokenizer=self.tokenizer,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=cpu_count(),
        )

        eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        eval_sampler = SequentialSampler(squad_dataset)
        eval_dataloader = DataLoader(squad_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation of QA *****")
        logger.info("  Num examples = %d", len(squad_dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        all_results = []

        for batch in tqdm(eval_dataloader, desc="Evaluating reader"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if self.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm self.models
                    if hasattr(self.model, "config") and hasattr(self.model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * self.lang_id).to(self.device)}
                        )

                outputs = self.model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = squad_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        # Compute predictions
        output_prediction_file = os.path.join(self.model_name_or_path, "predictions.json")
        output_nbest_file = os.path.join(self.model_name_or_path, "nbest_predictions.json")

        if True:
            output_null_log_odds_file = os.path.join(self.model_name_or_path, "null_odds.json")
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure

        if self.model_type in ["xlnet", "xlm"]:
            start_n_top = self.model.config.start_n_top if hasattr(self.model, "config") else self.model.module.config.start_n_top
            end_n_top = self.model.config.end_n_top if hasattr(self.model, "config") else self.model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                squad_examples,
                squad_features,
                all_results,
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                output_prediction_file=output_prediction_file,
                output_nbest_file=output_nbest_file,
                output_null_log_odds_file=output_null_log_odds_file,
                start_n_top=start_n_top,
                end_n_top=end_n_top,
                version_2_with_negative=True,
                tokenizer=self.okenizer,
                verbose_logging=True,
            )
        else:
            predictions = compute_predictions_logits(
                squad_examples,
                squad_features,
                all_results,
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                do_lower_case=True,
                output_prediction_file=output_prediction_file,
                output_nbest_file=output_nbest_file,
                output_null_log_odds_file=output_null_log_odds_file,
                verbose_logging=True,
                version_2_with_negative=True,
                null_score_diff_threshold=0.0,
                tokenizer=self.tokenizer,
            )
        logger.info('predictions: {}'.format(predictions))
        with open(output_nbest_file) as f:
            output_nbest = json.load(f)
        return output_nbest

    def answer_questions(self, ranked_questions_examples):
        return [self.answer_question(ranked_examples) for ranked_examples in ranked_questions_examples]

