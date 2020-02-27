from transformers.data.processors.glue import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          BertConfig, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification,
                          XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
                          AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                          XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
                          FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer)
from transformers.data.processors.glue import InputExample
from tqdm import tqdm
import numpy as np
import torch
import os
import logging
logger = logging.getLogger(__name__)

RANK_MODEL_CLASSES = {
    "distil-bert-cls": (BertConfig, DistilBertForSequenceClassification, BertTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

class TransformerRanker(object):
    def __init__(self, model_type='roberta', model_name_or_path='/net/csefiles/siemens/yzhang952/erenup/transformers-old/models_roberta/para_cls_250_10_2_semantic_negtive_large_bs256',
                 no_cuda=False, label_list=['0', '1'], output_mode='classification', per_gpu_eval_batch_size=2):
        assert model_type in RANK_MODEL_CLASSES.keys(), "model type: ".format(RANK_MODEL_CLASSES.keys())
        assert os.path.isdir(model_name_or_path), 'model_name_or_path should be a folder.'

        # Setup CUDA, GPU & distributed training
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()


        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.label_list = label_list
        self.output_mode = output_mode
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        num_labels = len(label_list)
        config_class, model_class, tokenizer_class = RANK_MODEL_CLASSES[model_type]
        self.config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task='ranker')
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True,)
        self.model = model_class.from_pretrained(model_name_or_path,
                                                 from_tf=bool(".ckpt" in model_name_or_path), config=self.config)
        self.model.to(self.device)
    def rank_question_examples(self, examples, n_docs=5):
        ranker_examples = [InputExample(guid=x['id'], text_a=x['question'], text_b=x['document'])
                           for x in examples]
        features = convert_examples_to_features(
            ranker_examples,
            self.tokenizer,
            label_list=['0', '1'],
            max_length=250,
            output_mode=self.output_mode,
            pad_on_left=bool(self.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
            task=None
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        eval_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
        eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        if self.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        preds_logits = None
        for batch in tqdm(eval_dataloader, desc="Evaluating ranker"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if self.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                logits = outputs[0]
            if preds_logits is None:
                preds_logits = logits.detach().cpu().numpy()
            else:
                preds_logits = np.append(preds_logits, logits.detach().cpu().numpy(), axis=0)
        if self.output_mode == "classification":
            preds = np.argmax(preds_logits, axis=1)
        elif self.output_mode == "regression":
            preds = np.squeeze(preds_logits)
        preds_logits = [x[1] for x in preds_logits.tolist()]
        for example, preds_logit in zip(examples, preds_logits):
            example['score'] = preds_logit

        examples = sorted(examples, key=lambda x: x['score'], reverse=True)
        examples = examples[:n_docs]
        return examples

    def rank_questions_examples(self, questions_examples, n_docs=5):
        return [self.rank_question_examples(examples, n_docs=n_docs) for examples in questions_examples]