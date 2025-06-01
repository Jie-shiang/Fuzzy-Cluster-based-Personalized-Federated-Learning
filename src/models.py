#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F

import torch
import time
from transformers import Data2VecAudioModel
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioPreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Data2VecAudioConfig, Wav2Vec2Processor

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

import numpy as np
import scipy
import copy

DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            <Tip warning={true}>
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [data2vec-audio-base](https://huggingface.co/facebook/data2vec-audio-base-960h), `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.
            </Tip>
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

_PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"

_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95

_CONFIG_FOR_DOC = "Data2VecAudioConfig"

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Basic feature preparation
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # MODIFIED: Use generic labels instead of dementia-specific
        labels = [{"labels": feature.get("labels", feature.get("dementia_labels", 0))} for feature in features]
        
        # Handle sample weights (new addition)
        sample_weights = None
        if "sample_weights" in features[0]:
            sample_weights = torch.tensor([f["sample_weights"] for f in features], dtype=torch.float32)
        
        # Process input features padding
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Process label features padding
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Handle labels mask
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        # MODIFIED: Use generic label key
        batch["class_labels"] = torch.tensor([torch.tensor(d.get('labels', d.get('dementia_labels', 0))) for d in labels])
        
        # Add sample weights to batch (new addition)
        if sample_weights is not None:
            batch["sample_weights"] = sample_weights
        
        # Handle fix_logits (if exists)
        if "fix_logits" in features[0].keys():
            fix_logits = [{"fix_logits": feature["fix_logits"]} for feature in features]
            batch["fix_logits"] = torch.tensor([[[torch.tensor(d) for d in item] for item in logit] 
                                              for fix_logit in fix_logits 
                                              for logit in fix_logit["fix_logits"]])
        
        return batch
    
def get_entropy(inputs_prob):
    """
    Calculate entropy of input probabilities
    Args:
        inputs_prob: numpy array or torch tensor
    Returns:
        batch_entropy: average entropy per sample
    """
    try:
        # Get input dimensions
        if isinstance(inputs_prob, np.ndarray):
            time_step, batch_size, vocab_size = inputs_prob.shape
        elif torch.is_tensor(inputs_prob):
            time_step, batch_size, vocab_size = inputs_prob.size()
        else:
            raise ValueError(f"Unsupported input type: {type(inputs_prob)}")
        
        batch_entropy = []
        for i in range(batch_size):
            entropy_sum = 0
            for j in range(time_step):
                prob = inputs_prob[j][i]
                if torch.is_tensor(prob):
                    prob = prob.cpu().detach().numpy()
                if not isinstance(prob, np.ndarray):
                    prob = np.array(prob)
                entropy_sum += scipy.stats.entropy(prob, base=None)
            batch_entropy.append(entropy_sum / (j+1))
        
        return batch_entropy
    except Exception as e:
        print(f"Warning: Error in entropy calculation: {str(e)}")
        return [0.0] * batch_size  # Return default values

def prox_loss(model1: nn.Module, model2: nn.Module):
    prox_loss_ = 0
    for i, (w, w_t) in enumerate(zip(model1.parameters(), model2.parameters())):
        #if i in [0,1,2,3,4,5]:
        prox_loss_ += (w-w_t).norm(2)

    if torch.is_tensor(prox_loss_):
        loss = prox_loss_.item()
    else:
        loss = prox_loss_
    return loss

class Data2VecAudioForCTC_CPFL(Data2VecAudioPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.args = args
        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.STAGE=args.STAGE                                                    # current stage
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output prob of each character

        # FedProx
        if args.FL_type == 2:                                                    # FedProx: save global model for loss
            print("Performing FedProx...")
            self.data2vec_audio_t = copy.deepcopy(self.data2vec_audio)
            self.dropout_t = copy.deepcopy(self.dropout)
            self.lm_head_t = copy.deepcopy(self.lm_head)
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if args.STAGE == 0:                                                      # train ASR encoder & decoder
            print("Current stage: 0")    
        elif args.STAGE == 1:                                                    # train ASR decoder alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
        
    # 1. LM_logit2loss function
    def LM_logit2loss(self, logits, labels, input_values, attention_mask, EXTRACT, sample_weights=None):
        """
        Calculate loss and entropy, supports sample weights
        Args:
            logits: model output logits
            labels: target labels
            input_values: input values
            attention_mask: attention mask
            EXTRACT: whether to calculate entropy
            sample_weights: sample weights (from membership values)
        Returns:
            loss: weighted CTC loss
            batch_entropy: entropy (if EXTRACT=True)
        """
        # Initialize batch_entropy
        batch_entropy = None
        
        # Calculate log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

        # Check labels
        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        # Handle attention mask
        attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

        # Handle labels
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)

        # Calculate loss
        with torch.backends.cudnn.flags(enabled=False):
            if sample_weights is not None:
                # Calculate loss per sample
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction='none',  # No reduction
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # Apply weights
                weighted_loss = (loss * sample_weights).sum() / sample_weights.sum()
                loss = weighted_loss
            else:
                # Use original reduction method
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            # Calculate entropy (if needed)
            if EXTRACT:
                try:
                    if torch.is_tensor(log_probs):
                        probs = torch.exp(log_probs)
                        probs_np = probs.detach().cpu().numpy()
                    else:
                        probs_np = np.exp(log_probs)
                    batch_entropy = get_entropy(probs_np)
                except Exception as e:
                    print(f"Warning: Failed to compute entropy: {str(e)}")
                    batch_entropy = None

        return loss, batch_entropy
    
    def get_encoder_attention(self, encoder_attention):

        """
        if encoder_attention is None:
            # MODIFIED: Use configurable time_steps_median instead of hardcoded value
            time_steps_median = getattr(self.args, 'time_steps_median', 130)  # Default to 130
            encoder_attention_1D = np.zeros(time_steps_median, dtype=np.float32)
            return encoder_attention_1D
        """
    
        # outputs[-1] # [24, batch_size, 16, time-step, time-step]
        encoder_attention = encoder_attention[-1][0][-1][:][:] # for batch_size=1: [time-step, time-step] from last layer's last head

        if torch.is_tensor(encoder_attention):
            encoder_attention = encoder_attention.cpu().detach().numpy()
        encoder_attention = np.asarray(encoder_attention) 
        #print(encoder_attention.shape) # [time-step, time-step]
        time_step, _ = encoder_attention.shape

        # MODIFIED: Use configurable median values instead of hardcoded
        if self.args.training_type == 1: # supervised
            time_steps_median = getattr(self.args, 'time_steps_median_supervised', 130)  # Default to 130
        else: # 2 dataset combined
            time_steps_median = getattr(self.args, 'time_steps_median_combined', 149)   # Default to 149

        # fill to same size
        if time_step < time_steps_median: # fill 0s
            new_shape = (int(time_steps_median), int(time_steps_median))
            new_arr = np.zeros(new_shape, dtype=encoder_attention.dtype) # array w/ all 0s
            new_arr[:time_step, :time_step] = encoder_attention         # first time_step*time_step is encoder_attention
        elif time_step > time_steps_median:
            new_arr = encoder_attention[:int(time_steps_median), :int(time_steps_median)]
                                                                        # clip to [time_steps_median, time_steps_median]
        else:
            new_arr = encoder_attention

        
        # to 1D
        axis_idx = 0 # perform on dim 0
        compress_type = "max" # can be var, mean, min, max, median

        if compress_type == "var":
            encoder_attention_1D = np.var(new_arr, axis=axis_idx)
        elif compress_type == "mean":
            encoder_attention_1D = np.mean(new_arr, axis=axis_idx)
        elif compress_type == "min":
            encoder_attention_1D = np.min(new_arr, axis=axis_idx)
        elif compress_type == "max":
            encoder_attention_1D = np.max(new_arr, axis=axis_idx)
        elif compress_type == "median":
            encoder_attention_1D = np.median(new_arr, axis=axis_idx)
        elif compress_type == "flat":
            encoder_attention_1D = np.array([item for sublist in new_arr for item in sublist])
        #print("encoder_attention_1D.shape: ", encoder_attention_1D.shape)
        
        return encoder_attention_1D
    
    # 2. forward function
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        class_labels=None,  # MODIFIED: Use generic class_labels instead of dementia_labels
        fix_logits=None,
        EXTRACT=False,
        sample_weights=None  # New parameter
    ):
        """
        Model forward pass
        Added sample_weights parameter for weighted training support
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        encoder_attention_1D = self.get_encoder_attention(outputs[-1])
        logits = self.lm_head(hidden_states)

        final_loss = None
        if (labels is not None) and (labels.numel() != 0):
            # Add sample_weights parameter
            final_loss, batch_entropy = self.LM_logit2loss(
                logits=logits,
                labels=labels,
                input_values=input_values,
                attention_mask=attention_mask,
                EXTRACT=EXTRACT,
                sample_weights=sample_weights
            )

            if self.args.FL_type == 2:  # FedProx
                final_loss = final_loss + self.args.mu/2 * prox_loss(self.data2vec_audio, self.data2vec_audio_t) \
                                    + self.args.mu/2 * prox_loss(self.dropout, self.dropout_t) \
                                    + self.args.mu/2 * prox_loss(self.lm_head, self.lm_head_t)
            elif self.args.FL_type == 3:  # FML
                KLdiv = nn.KLDivLoss(reduction='batchmean')
                log_prob = torch.log(F.softmax(logits, dim=2))
                fix_log_prob = torch.log(F.softmax(fix_logits, dim=2))
                kl_loss = KLdiv(log_prob, fix_log_prob)

                if self.args.FML_model == 0:
                    FML_weight = self.args.alpha
                elif self.args.FML_model == 1:
                    FML_weight = self.args.beta
                final_loss = FML_weight * final_loss + (1-FML_weight) * kl_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        if EXTRACT:
            hidden_states_mean = torch.mean(hidden_states, dim=1)
            logits_all = {
                'ASR logits': logits,
                'hidden_states': hidden_states,
                'hidden_states_mean': hidden_states_mean,
                'hidden_states_all': outputs.hidden_states,
                'loss': final_loss,
                'entropy': batch_entropy,
                'encoder_attention_1D': encoder_attention_1D
            }
        else:
            logits_all = logits

        return CausalLMOutput(
            loss=final_loss,
            logits=logits_all,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )