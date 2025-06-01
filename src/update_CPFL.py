def map_to_result(batch, processor, model, idx):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        labels = torch.tensor(batch["labels"]).unsqueeze(0)  
        logits = model(input_values, labels=labels, EXTRACT=True).logits                        # includes ASR logits, hidden_states_mean, loss
                                                                                                # output_attentions=True,
        asr_lg = logits['ASR logits']
    
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]                                     # predicted transcript
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)                       # ground truth transcript
    
    hidden_states_mean = logits["hidden_states_mean"].tolist()                                  # [batch_size, hidden_size]
    
    # compute freq. per character, and sort from largest to smallest
    flatten_arr = [item for sublist in pred_ids.numpy() for item in sublist]
    counter = Counter(flatten_arr)
    sorted_counter = counter.most_common()                                                      # sort from largest to smallest

    vocab_ratio_rank = [0] * 32                                                                 # initialize
    i = 0
    for num, count in sorted_counter:                                                           # num: char id，count: number of occurrence
        vocab_ratio_rank[i] = count / len(flatten_arr)                                          # convert to "ratio" (or freq.)
        i += 1                                                                                  # move to next char

    # replace inf and nan with 999
    df = pd.DataFrame([logits["loss"].tolist()])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    loss = df.values.tolist()                                                                   # [batch_size, 1]

    entropy = [logits["entropy"]]                                                               # [batch_size, 1]

    encoder_attention_1D = [logits["encoder_attention_1D"]]

    # MODIFIED: Use generic labels instead of dementia-specific
    df = pd.DataFrame({'path': batch["path"],                                                   # to know which sample
                    'text': batch["text"],
                    'class_labels': batch.get("class_labels", batch.get("dementia_labels", 0)),
                    'pred_str': batch["pred_str"]},
                    index=[idx])
    return df, hidden_states_mean, loss, entropy, [vocab_ratio_rank], encoder_attention_1D

def update_network_weight(args, source_path, target_weight, network):                           # update "network" in source_path with given weights
    # read source model                                                                         # return model   
    mask_time_prob = 0                                                                          # change config to avoid training stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                # use pre-trained config
    model = Data2VecAudioForCTC_CPFL.from_pretrained(args.pretrain_name, config=config, args=args)
                                                                                                # use pre-trained model
    model.config.ctc_zero_infinity = True                                                       # to avoid inf values

    if network == "ASR":                                                                        # given weight for ASR
        data2vec_audio, lm_head = target_weight

        model.data2vec_audio.load_state_dict(data2vec_audio)                                    # replace ASR encoder's weight
        model.lm_head.load_state_dict(lm_head)                                                  # replace ASR decoder's weight

    return copy.deepcopy(model)

class ASRLocalUpdate_CPFL(object):
    def __init__(self, args, dataset_supervised, global_test_dataset, client_id, cluster_id, model_in_path, model_out_path):
        self.args = args                                                                        # given configuration
        self.client_id = client_id                                                              # save client id
        self.cluster_id = cluster_id                                                            # save cluster id

        self.model_in_path = model_in_path                                                      # no info for client_id & global_round & cluster_id
        self.model_out_path = model_out_path   

        self.processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.device = 'cuda' if args.gpu else 'cpu'                                             # use gpu or cpu

        self.client_train_dataset_supervised=None
        self.ALL_client_train_dataset_supervised=None
        # if given dataset, get sub-dataset based on client_id & cluster_id
        if dataset_supervised is not None:
            self.client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, cluster_id)         # data of this client AND this cluster
            self.ALL_client_train_dataset_supervised, _ = train_split_supervised(args, dataset_supervised, client_id, None)           # data of this client       
            print("Dataset has ", len(self.client_train_dataset_supervised), " samples.")
        self.client_test_dataset = global_test_dataset                                          # global testing set
    
    def record_result(self, trainer, result_folder):                                            # save training loss, testing loss, and testing wer
        logger = SummaryWriter('./logs/' + result_folder.split("/")[-1])                        # use name of this model as folder's name

        for idx in range(len(trainer.state.log_history)):
            if "loss" in trainer.state.log_history[idx].keys():                                 # add in training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["loss"], trainer.state.log_history[idx]["epoch"]*100)

            elif "eval_loss" in trainer.state.log_history[idx].keys():                          # add in testing loss & WER, epoch*100 to obtain int
                logger.add_scalar('Loss/test', trainer.state.log_history[idx]["eval_loss"], trainer.state.log_history[idx]["epoch"]*100)
                logger.add_scalar('wer/test', trainer.state.log_history[idx]["eval_wer"], trainer.state.log_history[idx]["epoch"]*100)

            elif "train_loss" in trainer.state.log_history[idx].keys():                         # add in final training loss, epoch*100 to obtain int
                logger.add_scalar('Loss/train', trainer.state.log_history[idx]["train_loss"], trainer.state.log_history[idx]["epoch"]*100)
        logger.close()

    def model_train(self, model, client_train_dataset, save_path, num_train_epochs):
        """
        Training function with membership weighted training support
        """
        model.train()
        
        # Prepare training data and weights
        if self.args.use_soft_clustering and self.cluster_id is not None:
            # Get normalized memberships
            try:
                sample_weights = torch.tensor([
                    example['normalized_memberships'][self.cluster_id] 
                    for example in client_train_dataset
                ], dtype=torch.float32).to('cuda')
            except Exception as e:
                print(f"Warning: Failed to get memberships: {str(e)}")
                sample_weights = None
        else:
            sample_weights = None

        # Prepare DataCollator
        class WeightedDataCollator(DataCollatorCTCWithPadding):
            def __call__(self, features):
                batch = super().__call__(features)
                if 'sample_weights' in features[0]:
                    batch['sample_weights'] = torch.tensor(
                        [f['sample_weights'] for f in features],
                        dtype=torch.float32
                    ).to(batch['input_values'].device)
                return batch

        # Add weights to dataset
        if sample_weights is not None:
            weighted_dataset = client_train_dataset.map(
                lambda example, idx: {
                    **example, 
                    'sample_weights': sample_weights[idx].item()
                },
                with_indices=True
            )
        else:
            weighted_dataset = client_train_dataset

        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_path,
            group_by_length=True,
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            evaluation_strategy="steps",
            num_train_epochs=num_train_epochs,
            fp16=True,
            gradient_checkpointing=True, 
            save_steps=500,
            eval_steps=self.args.eval_steps,
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=1,
            log_level='debug',
            logging_strategy="steps",
        )

        # Create trainer
        trainer = CustomTrainer(
            model=model,
            data_collator=WeightedDataCollator(processor=self.processor, padding=True),
            args=training_args,
            compute_metrics=create_compute_metrics(self.processor),
            train_dataset=weighted_dataset,
            eval_dataset=self.client_test_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        # Print training info
        if self.cluster_id is not None:
            print(f" | Client {self.client_id} cluster {self.cluster_id} ready to train! |")
        else:
            print(f" | Client {self.client_id} ready to train! |")

        # Execute training
        trainer.train()

        # Handle results
        if self.args.STAGE == 1:  # freeze all, train ASR decoder alone
            torch.save(copy.deepcopy(trainer.model.lm_head.state_dict()), 
                    save_path + "/decoder_weights.pth")
            return_weights = [
                copy.deepcopy(trainer.model.data2vec_audio.state_dict()),
                copy.deepcopy(trainer.model.lm_head.state_dict())
            ]
            result_model = trainer.model
        else:
            trainer.save_model(save_path + "/final")
            return_weights = [
                copy.deepcopy(trainer.model.data2vec_audio.state_dict()),
                copy.deepcopy(trainer.model.lm_head.state_dict())
            ]
            result_model = trainer.model

        # Record training results
        self.record_result(trainer, save_path)

        return return_weights, result_model
    
    def gen_addLogit_fn(self, model_global):
        def map_to_logit(batch):
            with torch.no_grad():
                model = copy.deepcopy(model_global)
                input_values = torch.tensor(batch["input_values"]).unsqueeze(0).to("cuda")
                model = model.to("cuda")
                output = model(input_values)
                # Check output format and correctly extract logits
                logits = output.logits if not isinstance(output.logits, dict) else output.logits['ASR logits']
                batch["fix_logits"] = logits
            return batch
        return map_to_logit

    def update_weights(self, global_weights, global_round):
        # load training model
        if self.args.FL_type != 3:
            if global_weights == None:                                                              # train from model from model_in_path
                mask_time_prob = 0                                                                  # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
                model = load_model(self.args, self.model_in_path[:-7], config)
                model.config.ctc_zero_infinity = True                                               # to avoid inf values
            else:                                                                                   # update train model using given weight
                model = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                    # from model from model_in_path, update ASR's weight          
        elif self.args.FL_type == 3:                                                                # FML
            # initial local model
            mask_time_prob = 0                                                                      # change config to avoid training stopping
            config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
            self.args.FML_model = 0                                                                 # 0 for local --> alpha for local

            path = self.model_in_path[:-7] + "_localModel/"
            if os.path.exists(path):                                                                # if local file exits
                model_local = load_model(self.args, path[:-1], config)                              # load local model
            else:
                model_local = load_model(self.args, self.model_in_path[:-7], config)                # or use the same as mutual
            model_local.config.ctc_zero_infinity = True                                             # to avoid inf values                                                                                                    

            # load mutual
            self.args.FML_model = 1                                                                 # 1 for mutual --> beta for mutual
            
            if global_weights == None:                                                              # train from model from model_in_path
                mask_time_prob = 0                                                                  # change config to avoid training stopping
                config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
                                                                                                    # use pre-trained config
                model_mutual = load_model(self.args, self.model_in_path[:-7], config)
                model_mutual.config.ctc_zero_infinity = True                                        # to avoid inf values
            else:                                                                                   # update train model using given weight
                model_mutual = update_network_weight(args=self.args, source_path=self.model_in_path, target_weight=global_weights, network="ASR")
                                                                                                    # from model from model_in_path, update ASR's weight                
        
        if self.client_id == "public":                                                              # train using public dataset
            save_path = self.model_out_path + "_global"
            if self.args.CBFL:
                dataset = self.ALL_client_train_dataset_supervised                                  # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
            num_training_samples = len(self.client_train_dataset_supervised)

        elif self.args.training_type == 1:                                                          # supervised
            # save path for trained model (mutual model for FML)
            save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round)
            if self.cluster_id != None:
                save_path += "_cluster" + str(self.cluster_id)
            save_path += "_Training" + "Address"

            # CBFL use all training data from all cluster to train
            if self.args.CBFL:
                dataset = self.ALL_client_train_dataset_supervised                                   # train with all client data
            else:
                dataset = self.client_train_dataset_supervised
            
            if self.args.FL_type == 3:                                                               # FML: train model_local & model_mutal, only return model_mutual
                # train model_mutual
                print("train model_mutual")
                self.args.FML_model = 1                                                              # 1 for mutual
                dataset_mutual = dataset.map(self.gen_addLogit_fn(model_local))                      # local model as reference
                return_weights, _ = self.model_train(model_mutual, dataset_mutual, save_path, num_train_epochs=self.args.local_ep)
                num_training_samples = len(self.client_train_dataset_supervised)

                # remove previous model if exists
                if global_round > 0:
                    save_path_pre = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path_pre += "_cluster" + str(self.cluster_id)
                    save_path_pre += "_Training" + "Address"
                    shutil.rmtree(save_path_pre)

                # train model_local, and keep in local
                save_path += "_localModel"
                print("train model_local")
                self.args.FML_model = 0                                                              # 0 for local

                dataset_local = dataset.map(self.gen_addLogit_fn(model_mutual))                      # mutual as reference
                self.model_train(model_local, dataset_local, save_path, num_train_epochs=self.args.local_ep)

                # remove previous model if exists
                if global_round > 0:
                    save_path_pre = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path_pre += "_cluster" + str(self.cluster_id)
                    save_path_pre += "_Training" + "Address_localModel"
                    shutil.rmtree(save_path_pre)
                #del model, model_local
            else:                                                                                    # train 1 model
                return_weights, _ = self.model_train(model, dataset, save_path, num_train_epochs=self.args.local_ep)
                num_training_samples = len(self.client_train_dataset_supervised)
                # remove previous model if exists
                if global_round > 0:
                    save_path = self.model_out_path + "_client" + str(self.client_id) + "_round" + str(global_round - 1)
                    if self.cluster_id != None:
                        save_path += "_cluster" + str(self.cluster_id)
                    save_path += "_Training" + "Address"
                    shutil.rmtree(save_path)
        else:
            print("other training_type, such as type ", self.args.training_type, " not implemented yet")

        return return_weights, num_training_samples                                                  # return weight

    def extract_embs(self, TEST):
        # load model
        mask_time_prob = 0
        config = Data2VecAudioConfig.from_pretrained(self.args.pretrain_name, mask_time_prob=mask_time_prob)
        model = load_model(self.args, self.model_in_path, config)
        processor = self.processor

        if TEST:
            # get embeddings... 1 sample by 1 sample for client test
            df, hidden_states_mean, loss, entropy, vocab_ratio_rank, _, speech_features = map_to_result_MMSE(self.client_test_dataset[0], processor, model, 0)
            
            for i in range(len(self.client_test_dataset) - 1):
                df2, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, _, speech_features2 = map_to_result_MMSE(self.client_test_dataset[i+1], processor, model, i+1)
                df = pd.concat([df, df2], ignore_index=True)
                hidden_states_mean.extend(hidden_states_mean_2)
                loss.extend(loss2)
                entropy.extend(entropy2)
                vocab_ratio_rank.extend(vocab_ratio_rank2)
                speech_features.extend(speech_features2)
                print("\r"+ str(i), end="")

            return df, hidden_states_mean, loss, entropy, vocab_ratio_rank, speech_features

        else:
            hidden_states_mean_super = None
            loss_super = None
            entropy_super = None
            vocab_ratio_rank_super = None
            encoder_attention_1D_super = None
            speech_features_super = None

            # get embeddings... 1 sample by 1 sample for client train
            if (self.client_train_dataset_supervised != None) and (len(self.client_train_dataset_supervised) != 0):
                _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features = map_to_result_MMSE(self.client_train_dataset_supervised[0], processor, model, 0)
                
                for i in range(len(self.client_train_dataset_supervised) - 1):
                    _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2, speech_features2 = map_to_result_MMSE(self.client_train_dataset_supervised[i+1], processor, model, i+1)
                    hidden_states_mean.extend(hidden_states_mean_2)
                    loss.extend(loss2)
                    entropy.extend(entropy2)
                    vocab_ratio_rank.extend(vocab_ratio_rank2)
                    encoder_attention_1D.extend(encoder_attention_1D2)
                    speech_features.extend(speech_features2)
                    print("\r"+ str(i), end="")

                hidden_states_mean_super = hidden_states_mean
                loss_super = loss
                entropy_super = entropy
                vocab_ratio_rank_super = vocab_ratio_rank
                encoder_attention_1D_super = encoder_attention_1D
                speech_features_super = speech_features
            print("Training data Done")

            return hidden_states_mean_super, loss_super, entropy_super, vocab_ratio_rank_super, encoder_attention_1D_super, speech_features_super#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch

from transformers.training_args import TrainingArguments
from transformers import Trainer
from typing import Dict
import numpy as np
import os
import pandas as pd
from models import DataCollatorCTCWithPadding, Data2VecAudioForCTC_CPFL
from datasets import concatenate_datasets
import copy
from transformers import Data2VecAudioConfig, Wav2Vec2Processor
from tensorboardX import SummaryWriter
from utils import reorder_col, add_cluster_id, load_model
import pickle
import shutil
from utils import train_split_supervised

LOG_DIR = './logs/' #log/'

# REMOVED: Environment variables for hardcoded paths
# TODO: Update these paths based on your project structure
# CPFL_codeRoot = os.environ.get('CPFL_codeRoot')
# CPFL_dataRoot = os.environ.get('CPFL_dataRoot')
CPFL_codeRoot = './'  # UPDATE THIS PATH
CPFL_dataRoot = './data'  # UPDATE THIS PATH

from datasets import load_metric
wer_metric = load_metric("wer")
def create_compute_metrics(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    return compute_metrics

class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

from collections import Counter

# All available features list
ALL_AVAILABLE_FEATURES = {
   # Basic acoustic features
   "voiced_rate": "Voice activity rate",
   "pause_rate": "Pause ratio",
   "word_rate": "Word rate",
   
   # Advanced acoustic features
   "f0_std": "Pitch variation",
   "energy_mean": "Energy mean",
   "energy_std": "Energy standard deviation",
   "mfcc_mean": "MFCC mean",
   
   # Pause-related features
   "pause_mean_length": "Average pause length",
   "pause_count": "Pause count",
   "speech_rate_variance": "Speech rate variance",
   
   # Hidden States features (can be layers 1-24)
   "hidden_states_X": "Layer X hidden states" # X can be 1-24
}

def extract_key_features(input_values, processor, transcript, score, hidden_states_all=None, hidden_states_mean=None, selected_features=None):
    """
    Extract selected speech features
    Args:
        input_values: Audio input
        processor: Audio processor
        transcript: Text transcript
        score: Clinical score (generic instead of MMSE-specific)
        hidden_states_all: All layer hidden states
        hidden_states_mean: Mean hidden states
        selected_features: List of features to extract
    Returns:
        - numpy array: Dimension is len(basic features) + (if has hidden_states then +1024)
    """
    import librosa
    import numpy as np
    import torch
    
    features = {}
    
    # Prepare audio data
    if isinstance(input_values, torch.Tensor):
        audio_signal = input_values.squeeze().numpy()
    else:
        audio_signal = np.array(input_values).squeeze()
           
    sample_rate = processor.feature_extractor.sampling_rate
    duration = len(audio_signal) / sample_rate

    def check_nan(feature_name, value):
        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                print(f"NaN found in {feature_name}:")
                print(f"Shape: {value.shape}")
                print(f"NaN positions: {np.where(np.isnan(value))}")
                print(f"Value: {value}")
                return True
        else:
            if np.isnan(value):
                print(f"NaN found in {feature_name}:")
                print(f"Value: {value}")
                return True
        return False

    try:
        # 1. Basic acoustic features
        if "voiced_rate" in selected_features:
            voiced_frames = librosa.zero_crossings(audio_signal, pad=False)
            voiced_rate = np.sum(voiced_frames) / (2 * duration)
            if check_nan("voiced_rate", voiced_rate):
                features["voiced_rate"] = 0.0
            else:
                features["voiced_rate"] = float(voiced_rate)
       
        if "pause_rate" in selected_features:
            rms = librosa.feature.rms(y=audio_signal)[0]
            silence_threshold = np.mean(rms) * 0.5
            pauses = rms < silence_threshold
            pause_rate = np.sum(pauses) / len(pauses)
            if check_nan("pause_rate", pause_rate):
                features["pause_rate"] = 0.0
            else:
                features["pause_rate"] = float(pause_rate)
       
        if "word_rate" in selected_features:
            words = transcript.split()
            word_rate = len(words) / duration if duration > 0 else 0
            if check_nan("word_rate", word_rate):
                features["word_rate"] = 0.0
            else:
                features["word_rate"] = float(word_rate)
       
        if "clinical_score" in selected_features:  # MODIFIED: Generic score instead of MMSE
            features["clinical_score"] = float(score) if score is not None else 0.0

        # 2. Advanced acoustic features
        if "f0_std" in selected_features:
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    audio_signal,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate,
                    frame_length=2048,
                    win_length=1024,
                    center=True
                )
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    f0_std = np.std(valid_f0)
                    f0_std = np.clip(f0_std / 100.0, 0, 1)
                    features["f0_std"] = float(f0_std)
                else:
                    features["f0_std"] = 0.0
            except Exception as e:
                print(f"Error in f0_std extraction: {str(e)}")
                features["f0_std"] = 0.0

        if "mfcc_mean" in selected_features:
            try:
                mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                if check_nan("mfcc_mean", mfcc_means):
                    features["mfcc_mean"] = np.zeros(13, dtype=np.float32)
                else:
                    features["mfcc_mean"] = mfcc_means.astype(np.float32)
            except Exception as e:
                print(f"Error in mfcc_mean extraction: {str(e)}")
                features["mfcc_mean"] = np.zeros(13, dtype=np.float32)

        if "energy_mean" in selected_features:
            try:
                energy = librosa.feature.rms(y=audio_signal)[0]
                energy_mean = np.mean(energy)
                if check_nan("energy_mean", energy_mean):
                    features["energy_mean"] = 0.0
                else:
                    features["energy_mean"] = float(energy_mean)
            except Exception as e:
                print(f"Error in energy_mean extraction: {str(e)}")
                features["energy_mean"] = 0.0

        if "energy_std" in selected_features:
            try:
                if "energy_mean" not in features:
                    energy = librosa.feature.rms(y=audio_signal)[0]
                energy_std = np.std(energy)
                if check_nan("energy_std", energy_std):
                    features["energy_std"] = 0.0
                else:
                    features["energy_std"] = float(energy_std)
            except Exception as e:
                print(f"Error in energy_std extraction: {str(e)}")
                features["energy_std"] = 0.0

        # 3. Pause-related features
        if "pause_mean_length" in selected_features or "pause_count" in selected_features:
            try:
                if "pause_rate" not in features:
                    rms = librosa.feature.rms(y=audio_signal)[0]
                    silence_threshold = np.mean(rms) * 0.5
                    pauses = rms < silence_threshold
                pause_durations = np.diff(np.where(np.concatenate(([pauses[0]], 
                                                                pauses[1:] != pauses[:-1],
                                                                [True])))[0])[::2]
                
                if "pause_mean_length" in selected_features:
                    pause_mean = np.mean(pause_durations) if len(pause_durations) > 0 else 0
                    if check_nan("pause_mean_length", pause_mean):
                        features["pause_mean_length"] = 0.0
                    else:
                        features["pause_mean_length"] = float(pause_mean)
                
                if "pause_count" in selected_features:
                    features["pause_count"] = float(len(pause_durations))
            except Exception as e:
                print(f"Error in pause features extraction: {str(e)}")
                if "pause_mean_length" in selected_features:
                    features["pause_mean_length"] = 0.0
                if "pause_count" in selected_features:
                    features["pause_count"] = 0.0

        # 4. Speech rate variance
        if "speech_rate_variance" in selected_features:
            try:
                if len(transcript.split()) > 1:
                    word_durations = [len(word) for word in transcript.split()]
                    rate_var = np.var(word_durations)
                    if check_nan("speech_rate_variance", rate_var):
                        features["speech_rate_variance"] = 0.0
                    else:
                        features["speech_rate_variance"] = float(rate_var)
                else:
                    features["speech_rate_variance"] = 0.0
            except Exception as e:
                print(f"Error in speech_rate_variance extraction: {str(e)}")
                features["speech_rate_variance"] = 0.0

    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        # If error occurs, set default values for all selected basic features
        for feat in selected_features:
            if feat not in features and feat != "hidden_states_mean":
                features[feat] = 0.0

    # Build final feature vector
    feature_values = []
    
    # 1. First process all non-hidden states features
    for feat in selected_features:
        if not (feat.startswith("hidden_states_") or feat == "hidden_states_mean"):
            if feat == "mfcc_mean":
                # mfcc_mean is 13-dimensional
                feature_values.extend(features.get(feat, np.zeros(13, dtype=np.float32)))
            else:
                # Other basic features are 1-dimensional
                feature_values.append(float(features.get(feat, 0.0)))
    
    # 2. Process hidden states features
    if "hidden_states_mean" in selected_features:
        if hidden_states_mean is not None:
            if isinstance(hidden_states_mean, torch.Tensor):
                hs_mean = hidden_states_mean.cpu().numpy()
            elif isinstance(hidden_states_mean, list):
                hs_mean = np.array(hidden_states_mean)
            else:
                hs_mean = hidden_states_mean
            
            # Ensure it's 1D
            hs_mean = hs_mean.flatten()
            
            # Ensure length is 1024
            if len(hs_mean) != 1024:
                print(f"Warning: hidden_states_mean unexpected length: {len(hs_mean)}")
                hs_mean = np.zeros(1024)
            
            feature_values.extend(hs_mean)
        else:
            feature_values.extend(np.zeros(1024))
    
    # Process other hidden states layers
    for feat in selected_features:
        if feat.startswith("hidden_states_") and feat != "hidden_states_mean":
            try:
                layer_num = int(feat.split("_")[-1])
                layer_idx = layer_num - 1
                if hidden_states_all is not None and 0 <= layer_idx < len(hidden_states_all):
                    layer_states = hidden_states_all[layer_idx]
                    if isinstance(layer_states, torch.Tensor):
                        layer_mean = torch.mean(layer_states, dim=1).flatten()
                        feature_values.extend(layer_mean.cpu().numpy())
                    else:
                        feature_values.extend(np.array(layer_states).flatten())
                else:
                    feature_values.extend(np.zeros(1024))
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to extract {feat}, error: {str(e)}")
                feature_values.extend(np.zeros(1024))

    # Convert to numpy array and ensure type is float32
    final_array = np.array(feature_values, dtype=np.float32)
    
    # Validate dimensions
    expected_dim = sum([
        # Dimension of basic features
        sum(13 if f == "mfcc_mean" else 1 
            for f in selected_features 
            if not (f.startswith("hidden_states_") or f == "hidden_states_mean")),
        # Dimension of hidden states
        sum(1024 for f in selected_features 
            if f.startswith("hidden_states_") or f == "hidden_states_mean")
    ])
    
    actual_dim = len(final_array)
    if expected_dim != actual_dim:
        print(f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
        print(f"Selected features: {selected_features}")
        
    return final_array

def map_to_result_MMSE(batch, processor, model, idx):
    # Define features to use
    selected_features = [
        "word_rate",
        ]

    # REMOVED: Hardcoded MMSE file path
    # TODO: Update this to your clinical score file path
    score_file = "./data/clinical_scores.txt"  # UPDATE THIS PATH
    
    # Extract speaker ID from path
    speaker_id = batch["path"].split('_')[0]
    
    # Find corresponding clinical score
    try:
        if os.path.exists(score_file):
            score_df = pd.read_csv(score_file, sep=';')
            score = score_df[score_df['ID'] == speaker_id]['score'].values[0]
            if score == 'NA':
                score = None
            else:
                score = float(score)
        else:
            print(f"Warning: Clinical score file not found: {score_file}")
            score = None
    except:
        score = None

    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        labels = torch.tensor(batch["labels"]).unsqueeze(0)  
        logits = model(input_values, labels=labels, EXTRACT=True).logits
        asr_lg = logits['ASR logits']
    
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    # Get hidden_states_mean, keeping original method
    hidden_states_mean = logits["hidden_states_mean"].tolist()
    hidden_states_all = logits.get('hidden_states_all', None)

    try:
        speech_features = extract_key_features(
            input_values=batch["input_values"],
            processor=processor,
            transcript=batch["pred_str"],
            score=score,  # MODIFIED: Generic score instead of MMSE
            hidden_states_all=hidden_states_all,
            hidden_states_mean=hidden_states_mean,
            selected_features=selected_features
        ).tolist()
    except Exception as e:
        print(f"Error in feature extraction for file {batch['path']}: {str(e)}")
        # Calculate correct default size
        default_size = sum(1024 if f in ["hidden_states_mean"] or f.startswith("hidden_states_") else 1 
                          for f in selected_features)
        speech_features = [0.0] * default_size

    # Calculate vocab ratio rank
    flatten_arr = [item for sublist in pred_ids.numpy() for item in sublist]
    counter = Counter(flatten_arr)
    sorted_counter = counter.most_common()

    vocab_ratio_rank = [0] * 32
    i = 0
    for num, count in sorted_counter:
        vocab_ratio_rank[i] = count / len(flatten_arr)
        i += 1

    # Handle loss
    df = pd.DataFrame([logits["loss"].tolist()])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    loss = df.values.tolist()

    entropy = [logits["entropy"]]
    encoder_attention_1D = [logits["encoder_attention_1D"]]

    # Basic features in DataFrame
    df_features = {}
    for i, feat in enumerate(selected_features):
        if not feat.startswith("hidden_states_"):  # Only include basic features
            try:
                df_features[feat] = speech_features[i]
            except IndexError:
                print(f"Warning: Could not get feature {feat} at index {i}")
                df_features[feat] = 0.0
    
    # Build DataFrame
    try:
        # MODIFIED: Use generic labels instead of dementia-specific
        df = pd.DataFrame({
            'path': batch["path"],
            'text': batch["text"],
            'class_labels': batch.get("class_labels", batch.get("dementia_labels", 0)),  # Generic labels
            'pred_str': batch["pred_str"],
            **df_features
        }, index=[idx])
    except Exception as e:
        print(f"Error creating DataFrame: {str(e)}")
        df = pd.DataFrame({
            'path': batch["path"],
            'text': batch["text"],
            'class_labels': batch.get("class_labels", batch.get("dementia_labels", 0)),
            'pred_str': batch["pred_str"]
        }, index=[idx])

    return df, hidden_states_mean, loss, entropy, [vocab_ratio_rank], encoder_attention_1D, [speech_features]