import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from transformers import (
        WEIGHTS_NAME,
        AdamW,
        AlbertConfig,
        AlbertForQuestionAnswering,
        AlbertTokenizer,
        BertConfig,
        BertForQuestionAnswering,
        BertTokenizer,
        DistilBertConfig,
        DistilBertForQuestionAnswering,
        DistilBertTokenizer,
        XLMConfig,
        XLMForQuestionAnswering,
        XLMTokenizer,
        XLNetConfig,
        XLNetForQuestionAnswering,
        XLNetTokenizer,
        get_linear_schedule_with_warmup,
        squad_convert_examples_to_features,
        )
from transformers.data.metrics.squad_metrics import (
        compute_predictions_log_probs,
        compute_predictions_logits,
        squad_evaluate,
        )

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

ALL_MODELS = sum(
        (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
        (),
        )

MODEL_CLASSES = {
        "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
        "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
        }

class Reader:

    def __init__(self, model_type, model_path, output_dir, batch_size=256, max_seq=384, max_answer_length=35, n_best_size=20, doc_stride=128, max_query_length=64, workers=2):
        self.model_type = model_type
        self.model_path = model_path
        self.output_dir= output_dir
        self.max_seq = max_seq
        self.max_answer_length= max_answer_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.workers = workers
        self.batch_size= batch_size
        self.n_best_size = n_best_size

    def load_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type] 

        self.model = model_class.from_pretrained(self.model_path)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_path)

        self.model.to(device)

    def evaluate(self, samples):
        if self.model is None or self.tokenizer is None:
            return []
        features, dl = self.__transform_to_features(samples)
        return self.__get_predictions(dl, features, samples)

    def __transform_to_features(self, samples):
        features, dataset = squad_convert_examples_to_features(
                examples=samples,
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq, 
                doc_stride=self.doc_stride, 
                max_query_length=self.max_query_length, 
                is_training=False, 
                return_dataset='pt',
                threads=self.workers)


        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return features, dataloader

    def __get_predictions(self, dataloader, features, samples, prefix=""):
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_results = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.iterableto(device) for t in batch)

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

                outputs = self.model(**inputs)

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
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

        #compute predictions

        output_prediction_file = os.path.join(self.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.output_dir, "nbest_predictions_{}.json".format(prefix))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None
        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = self.model.config.start_n_top if hasattr(self.model, "config") else self.model.module.config.start_n_top
            end_n_top = self.model.config.end_n_top if hasattr(self.model, "config") else self.model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                    samples,
                    features,
                    all_results,
                    self.n_best_size,
                    self.max_answer_length,
                    output_prediction_file,
                    output_nbest_file,
                    output_null_log_odds_file,
                    start_n_top,
                    end_n_top,
                    True,
                    self.tokenizer,
                    False,
                    )
        else:
            predictions = compute_predictions_logits(
                    samples,
                    features,
                    all_results,
                    self.n_best_size,
                    self.max_answer_length,
                    self.do_lower_case,
                    output_prediction_file,
                    output_nbest_file,
                    output_null_log_odds_file,
                    False,
                    True,
                    self.null_score_diff_threshold,
                    self.tokenizer,
                    )

            return predictions
