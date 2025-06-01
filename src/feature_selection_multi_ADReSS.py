import os
import glob
import logging
import argparse
import time
from datetime import timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import soundfile as sf
import skfuzzy as fuzz
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, mutual_info_regression

import torch
from transformers import Wav2Vec2Processor, Data2VecAudioModel, AutoFeatureExtractor

def setup_logger(log_path='feature_selection.log'):
    logger = logging.getLogger('feature_selection')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def compute_xb(u, centers, X, m=2.0):
    c, N = u.shape
    dist2 = np.zeros((c, N))
    for k in range(c):
        diff = X - centers[k]
        dist2[k] = np.sum(diff**2, axis=1)
    numerator = np.sum((u**m) * dist2)
    cd2 = np.sum((centers[:,None,:] - centers[None,:,:])**2, axis=2)
    np.fill_diagonal(cd2, np.inf)
    denominator = N * np.min(cd2)
    return numerator / denominator

def weighted_mutual_information(X_col, u, n_bins=10):
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    bins = kb.fit_transform(X_col.reshape(-1,1)).astype(int).flatten()
    c, N = u.shape
    W = np.sum(u)
    p_k = np.sum(u, axis=1) / W

    mi = 0.0
    for b in np.unique(bins):
        idx = np.where(bins==b)[0]
        if idx.size == 0: continue
        w_b = np.sum(u[:, idx])
        p_b = w_b / W
        for k in range(c):
            w_bk = np.sum(u[k, idx])
            if w_bk<=0: continue
            p_bk = w_bk / W
            mi += p_bk * np.log(p_bk / (p_b * p_k[k]))
    return mi

def evaluate_feature_pairs(X, u, selected_idxs, feature_names, logger):
    n_features = len(selected_idxs)
    pair_scores = np.zeros((n_features, n_features))
    y = np.argmax(u, axis=0)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            feat_i = selected_idxs[i]
            feat_j = selected_idxs[j]
            X_pair = X[:, [feat_i, feat_j]]
            
            clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='saga', C=1.0)
            score = cross_val_score(clf, X_pair, y, cv=5, scoring='accuracy').mean()
            
            logger.info(f"  Feature pair {feature_names[feat_i]} + {feature_names[feat_j]}: score = {score:.4f}")
            
            pair_scores[i, j] = score
            pair_scores[j, i] = score
    
    best_i, best_j = np.unravel_index(np.argmax(pair_scores), pair_scores.shape)
    best_i_idx = selected_idxs[best_i]
    best_j_idx = selected_idxs[best_j]
    
    return best_i_idx, best_j_idx, pair_scores

def feature_subset_selection(X, u, selected_idxs, feature_names, logger, max_subset_size=5):
    y = np.argmax(u, axis=0)
    best_scores = {}
    
    for size in range(1, min(max_subset_size + 1, len(selected_idxs) + 1)):
        logger.info(f"  Evaluating feature subsets of size {size}...")
        best_score_for_size = 0
        best_subset_for_size = []
        
        for subset_indices in combinations(range(len(selected_idxs)), size):
            feat_indices = [selected_idxs[i] for i in subset_indices]
            X_subset = X[:, feat_indices]
            feat_names = [feature_names[idx] for idx in feat_indices]
            
            clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='saga', C=1.0)
            score = cross_val_score(clf, X_subset, y, cv=5, scoring='accuracy').mean()
            
            if score > best_score_for_size:
                best_score_for_size = score
                best_subset_for_size = feat_indices
                logger.info(f"    New best subset size {size}: {feat_names} (score = {score:.4f})")
        
        best_scores[size] = (best_subset_for_size, best_score_for_size)
    
    overall_best_size = max(best_scores.keys(), key=lambda k: best_scores[k][1])
    overall_best_subset, overall_best_score = best_scores[overall_best_size]
    
    return best_scores, overall_best_subset, overall_best_score

def create_feature_interactions(X, selected_idxs, feature_names, logger):
    logger.info("Creating feature interactions...")
    
    interaction_features = []
    interaction_names = []
    interaction_idxs = []
    
    for i, j in combinations(range(len(selected_idxs)), 2):
        idx_i = selected_idxs[i]
        idx_j = selected_idxs[j]
        
        interaction = (X[:, idx_i] * X[:, idx_j]).reshape(-1, 1)
        interaction_features.append(interaction)
        
        name = f"{feature_names[idx_i]}*{feature_names[idx_j]}"
        interaction_names.append(name)
        interaction_idxs.append((idx_i, idx_j, "product"))
        
        logger.info(f"  Added interaction: {name}")
    
    for i in range(len(selected_idxs)):
        idx = selected_idxs[i]
        
        squared = np.square(X[:, idx]).reshape(-1, 1)
        interaction_features.append(squared)
        
        name = f"{feature_names[idx]}^2"
        interaction_names.append(name)
        interaction_idxs.append((idx, None, "square"))
        
        logger.info(f"  Added squared term: {name}")
    
    return interaction_features, interaction_names, interaction_idxs

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

def extract_hidden_states_mean(audio_signal, sample_rate, model_path, device, logger=None):
    """
    Extract hidden states from audio signal
    
    Args:
        audio_signal: Audio signal
        sample_rate: Sampling rate
        model_path: Model path
        device: Device to use ('cpu' or 'cuda:X')
        logger: Logger instance
        
    Returns:
        Hidden state feature vector
    """
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        
        # Use specified device
        if logger:
            logger.info(f"Using device: {device} for hidden state extraction")
            
        model = Data2VecAudioModel.from_pretrained(model_path).to(device)
        
        inputs = feature_extractor(audio_signal, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            
        all_hidden_states = outputs.hidden_states
        hidden_states = all_hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
        
        return hidden_states
        
    except RuntimeError as e:
        if "CUDA" in str(e) and "cpu" not in device:
            if logger:
                logger.warning(f"Device {device} CUDA error: {e}, falling back to CPU")
            # Fallback to CPU
            return extract_hidden_states_mean(audio_signal, sample_rate, model_path, 'cpu', logger)
        else:
            if logger:
                logger.error(f"Error extracting hidden states: {e}")
            raise

def extract_key_features(audio_signal, sample_rate, transcript, selected_features=None, model_path=None):
    import librosa
    import numpy as np
    
    features = {}
    
    if not isinstance(audio_signal, np.ndarray):
        audio_signal = np.array(audio_signal).squeeze()
    else:
        audio_signal = audio_signal.squeeze()
           
    duration = len(audio_signal) / sample_rate

    try:
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
                features["mfcc_mean"] = float(np.mean(mfcc_means))
                if check_nan("mfcc_mean", features["mfcc_mean"]):
                    features["mfcc_mean"] = 0.0
            except Exception as e:
                print(f"Error in mfcc_mean extraction: {str(e)}")
                features["mfcc_mean"] = 0.0

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
        for feat in selected_features:
            if feat not in features and feat != "hidden_states_mean":
                features[feat] = 0.0

    return features

def extract_and_process_hidden_states(audio_files, transcripts, sample_rates, model_path, device, n_components=10, logger=None):
    """
    Extract hidden states from multiple audio files and apply PCA for dimensionality reduction
    
    Args:
        audio_files: List of audio files
        transcripts: Corresponding text list
        sample_rates: Corresponding sampling rate list
        model_path: Model path
        device: Device to use ('cpu' or 'cuda:X')
        n_components: Dimensions after PCA reduction
        logger: Logger instance
        
    Returns:
        Dimensionality-reduced hidden state feature matrix
    """
    logger.info(f"Extracting hidden states from {len(audio_files)} files using device: {device}...")
    
    all_hidden_states = []
    error_count = 0
    max_retry = 3
    
    for i, (audio, sr, _) in enumerate(zip(audio_files, sample_rates, transcripts)):
        if i % 10 == 0:
            logger.info(f"  Processing file {i+1}/{len(audio_files)}")
        
        retry_count = 0
        success = False
        
        # Attempt to extract hidden states, retry if failed
        while not success and retry_count < max_retry:
            try:
                # First try using specified device
                current_device = device
                
                # If previous error, try other GPU or CPU
                if retry_count > 0:
                    if "cuda" in device and retry_count == 1:
                        # First retry: try CPU
                        current_device = "cpu"
                        logger.info(f"  File {i+1}: retry {retry_count}/{max_retry}, using CPU")
                    elif "cuda" in device and retry_count == 2:
                        # Second retry: try different GPU (if available)
                        if ":" in device:
                            gpu_id = int(device.split(":")[-1])
                            available_gpus = [idx for idx in range(torch.cuda.device_count()) 
                                             if idx != gpu_id]
                            if available_gpus:
                                current_device = f"cuda:{available_gpus[0]}"
                                logger.info(f"  File {i+1}: retry {retry_count}/{max_retry}, using another GPU: {current_device}")
                            else:
                                current_device = "cpu"
                                logger.info(f"  File {i+1}: retry {retry_count}/{max_retry}, no other GPU available, using CPU")
                        else:
                            current_device = "cpu"
                            logger.info(f"  File {i+1}: retry {retry_count}/{max_retry}, using CPU")
                
                hidden_states = extract_hidden_states_mean(audio, sr, model_path, current_device, logger)
                all_hidden_states.append(hidden_states)
                success = True
                
            except Exception as e:
                retry_count += 1
                logger.error(f"  File {i+1} hidden state extraction error (attempt {retry_count}/{max_retry}): {str(e)}")
                # If this is the last attempt, use zero vector
                if retry_count >= max_retry:
                    logger.warning(f"  File {i+1} all attempts failed, using zero vector")
                    all_hidden_states.append(np.zeros(768))
                    error_count += 1
                    
                # Clean GPU memory between retries
                clean_gpu_memory()
    
    if not all_hidden_states:
        logger.error("No hidden states extracted")
        return None
    
    if error_count > 0:
        logger.warning(f"Total {error_count}/{len(audio_files)} files failed extraction, using zero vectors")
    
    hidden_states_matrix = np.vstack(all_hidden_states)
    
    logger.info(f"Hidden states matrix shape: {hidden_states_matrix.shape}")
    logger.info(f"Using PCA to reduce dimensions to {n_components} components...")
    
    pca = PCA(n_components=n_components)
    reduced_hidden_states = pca.fit_transform(hidden_states_matrix)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    logger.info(f"Explained variance using {n_components} components: {explained_variance:.2f}%")
    
    return reduced_hidden_states

def clean_gpu_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def run_filter_feature_selection(X_scaled, best_u, feature_names, K_filter, M, theta, logger):
    """Use Filter method for feature selection"""
    logger.info("=== Using Filter method for feature selection ===")
    d = X_scaled.shape[1]
    countF = np.zeros(d)
    sumMI = np.zeros(d)
    
    for r in tqdm(range(1, M+1), desc="Running Filter feature selection"):
        logger.info(f"-- Iteration {r}/{M} --")
        
        MI = np.array([weighted_mutual_information(X_scaled[:,j], best_u)
                     for j in range(d)])
        topF = np.argsort(MI)[-K_filter:]
        for j in topF:
            countF[j] += 1
            sumMI[j] += MI[j]
        logger.info(f"  Filter picks: {[feature_names[j] for j in topF]}")
    
    stable_idxs = [j for j in range(d) if countF[j] / M >= theta]
    
    logger.info(f"Stable feature indices: {stable_idxs}")
    logger.info(f"Stable features: {[feature_names[j] for j in stable_idxs]}")
    
    if not stable_idxs:
        logger.warning("No stable features found with Filter method")
        return [], {}
    
    scores = {j: sumMI[j] / countF[j] if countF[j] > 0 else 0 for j in stable_idxs}
    
    return stable_idxs, scores

def run_rfe_feature_selection(X_scaled, best_u, feature_names, K_embedded, M, theta, logger):
    """Use RFE method for feature selection"""
    logger.info("=== Using RFE method for feature selection ===")
    d = X_scaled.shape[1]
    countR = np.zeros(d)
    sumImp = np.zeros(d)
    
    y = np.argmax(best_u, axis=0)  # Use maximum membership as hard labels
    
    for r in tqdm(range(1, M+1), desc="Running RFE feature selection"):
        logger.info(f"-- Iteration {r}/{M} --")
        
        # Use soft labels and sample weights
        sw = np.max(best_u, axis=0)  # Use maximum membership as sample weights
        
        # Use LogisticRegression with RFE
        clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)
        selector = RFE(clf, n_features_to_select=K_embedded, step=1)
        
        # Train RFE with sample weights
        selector.fit(X_scaled, y, sample_weight=sw)
        
        # Get selected features
        selected = selector.get_support()
        topR = np.where(selected)[0]
        
        # Calculate feature importance
        imp = np.zeros(d)
        imp[topR] = 1.0
        
        for j in topR:
            countR[j] += 1
            sumImp[j] += imp[j]
        
        logger.info(f"  RFE picks: {[feature_names[j] for j in topR]}")
    
    stable_idxs = [j for j in range(d) if countR[j] / M >= theta]
    
    logger.info(f"Stable feature indices: {stable_idxs}")
    logger.info(f"Stable features: {[feature_names[j] for j in stable_idxs]}")
    
    if not stable_idxs:
        logger.warning("No stable features found with RFE method")
        return [], {}
    
    scores = {j: sumImp[j] / countR[j] if countR[j] > 0 else 0 for j in stable_idxs}
    
    return stable_idxs, scores

def run_lasso_feature_selection(X_scaled, best_u, feature_names, K_embedded, M, theta, logger):
    """Use Lasso method for feature selection"""
    logger.info("=== Using Lasso method for feature selection ===")
    d = X_scaled.shape[1]
    countE = np.zeros(d)
    sumImp = np.zeros(d)
    
    for r in tqdm(range(1, M+1), desc="Running Lasso feature selection"):
        logger.info(f"-- Iteration {r}/{M} --")
        
        # Use Lasso model (L1 regularization) for feature selection
        X_e = np.vstack([X_scaled]*best_u.shape[0])
        y_e = np.concatenate([np.full(X_scaled.shape[0], k) for k in range(best_u.shape[0])])
        sw = best_u.flatten(order='C')
        
        # Use LogisticRegression with L1 regularization
        clf = LogisticRegression(
            penalty='l1', solver='saga', multi_class='multinomial',
            C=1.0, max_iter=1000
        )
        clf.fit(X_e, y_e, sample_weight=sw)
        W = clf.coef_
        Imp = np.sum(np.abs(W), axis=0)
        topE = np.argsort(Imp)[-K_embedded:]
        
        for j in topE:
            countE[j] += 1
            sumImp[j] += Imp[j]
            
        logger.info(f"  Lasso picks: {[feature_names[j] for j in topE]}")
    
    stable_idxs = [j for j in range(d) if countE[j] / M >= theta]
    
    logger.info(f"Stable feature indices: {stable_idxs}")
    logger.info(f"Stable features: {[feature_names[j] for j in stable_idxs]}")
    
    if not stable_idxs:
        logger.warning("No stable features found with Lasso method")
        return [], {}
    
    scores = {j: sumImp[j] / countE[j] if countE[j] > 0 else 0 for j in stable_idxs}
    
    return stable_idxs, scores

def run_filter_lasso_feature_selection(X_scaled, best_u, feature_names, K_filter, K_embedded, M, theta, logger):
    """Use Filter + Lasso method for feature selection"""
    logger.info("=== Using Filter + Lasso method for feature selection ===")
    d = X_scaled.shape[1]
    countF = np.zeros(d)
    countE = np.zeros(d)
    sumMI = np.zeros(d)
    sumImp = np.zeros(d)

    for r in tqdm(range(1, M+1), desc="Running Filter+Lasso feature selection"):
        logger.info(f"-- Iteration {r}/{M} --")
        
        # Filter part
        MI = np.array([weighted_mutual_information(X_scaled[:,j], best_u)
                     for j in range(d)])
        topF = np.argsort(MI)[-K_filter:]
        for j in topF:
            countF[j] += 1
            sumMI[j] += MI[j]
        logger.info(f"  Filter picks: {[feature_names[j] for j in topF]}")

        # Lasso part
        X_e = np.vstack([X_scaled]*best_u.shape[0])
        y_e = np.concatenate([np.full(X_scaled.shape[0], k) for k in range(best_u.shape[0])])
        sw = best_u.flatten(order='C')
        
        clf = LogisticRegression(
            penalty='l1', solver='saga', multi_class='multinomial',
            C=1.0, max_iter=1000
        )
        clf.fit(X_e, y_e, sample_weight=sw)
        W = clf.coef_
        Imp = np.sum(np.abs(W), axis=0)
        topE = np.argsort(Imp)[-K_embedded:]
        for j in topE:
            countE[j] += 1
            sumImp[j] += Imp[j]
        logger.info(f"  Lasso picks: {[feature_names[j] for j in topE]}")

    # Combine results from both methods
    stable_idxs = [j for j in range(d)
                 if max(countF[j], countE[j]) / M >= theta]
    
    logger.info(f"Stable feature indices: {stable_idxs}")
    logger.info(f"Stable features: {[feature_names[j] for j in stable_idxs]}")
    
    if not stable_idxs:
        logger.warning("No stable features found with Filter+Lasso method")
        return [], {}
    
    # Combine scores
    scores = {j: (sumMI[j] + sumImp[j]) / max(countF[j], countE[j])
            for j in stable_idxs}
    
    return stable_idxs, scores

def analyze_features(X_scaled, best_u, stable_idxs, scores, feature_names, logger, method_name):
    """Analyze features and generate report"""
    if not stable_idxs:
        logger.warning(f"No stable features to analyze for {method_name}")
        return None
    
    # Sort features
    top_features = sorted(scores, key=scores.get, reverse=True)[:min(5, len(scores))]
    top_features_names = [feature_names[j] for j in top_features]
    
    logger.info(f"*** {method_name} - Top {len(top_features)} features: {top_features_names} ***")
    print(f"{method_name} - Top {len(top_features)} features: {top_features_names}")
    
    # Check if hidden state components are selected
    hidden_comps_selected = [f for f in top_features_names if f.startswith("hidden_comp_")]
    if hidden_comps_selected:
        logger.info(f"{method_name} - Selected hidden state components: {hidden_comps_selected}")
        print(f"{method_name} - Selected hidden state components: {hidden_comps_selected}")
    
    single_feature_ranking = [(feature_names[j], scores[j]) for j in sorted(scores, key=scores.get, reverse=True)]
    
    # Analyze feature pairs
    if len(stable_idxs) >= 2:
        logger.info(f"{method_name} - Evaluating feature pairs...")
        best_pair_i, best_pair_j, pair_scores = evaluate_feature_pairs(
            X_scaled, best_u, stable_idxs, feature_names, logger
        )
        best_pair = (feature_names[best_pair_i], feature_names[best_pair_j])
        logger.info(f"{method_name} - Best feature pair: {best_pair[0]} + {best_pair[1]}")
        print(f"{method_name} - Best feature pair: {best_pair[0]} + {best_pair[1]}")
        
        X_best_pair = X_scaled[:, [best_pair_i, best_pair_j]]
        y = np.argmax(best_u, axis=0)
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
        best_pair_score = cross_val_score(clf, X_best_pair, y, cv=5, scoring='accuracy').mean()
        logger.info(f"{method_name} - Best feature pair classification accuracy: {best_pair_score:.4f}")
        print(f"{method_name} - Best feature pair classification accuracy: {best_pair_score:.4f}")
    
    # Analyze feature subsets
    subset_results = {}
    if len(stable_idxs) >= 2:
        logger.info(f"{method_name} - Finding best feature subsets...")
        best_scores, overall_best_subset, overall_best_score = feature_subset_selection(
            X_scaled, best_u, stable_idxs, feature_names, logger, max_subset_size=5
        )
        
        for size, (subset, score) in best_scores.items():
            subset_names = [feature_names[j] for j in subset]
            subset_results[size] = (subset_names, score)
            logger.info(f"{method_name} - Best subset size {size}: {subset_names} (score = {score:.4f})")
            print(f"{method_name} - Best subset size {size}: {subset_names} (score = {score:.4f})")
    
    # Analyze feature interactions
    if len(stable_idxs) >= 2:
        logger.info(f"{method_name} - Testing feature interactions...")
        interaction_features, interaction_names, interaction_idxs = create_feature_interactions(
            X_scaled, stable_idxs, feature_names, logger
        )
        
        X_extended = np.hstack([X_scaled] + [f for f in interaction_features])
        extended_feature_names = feature_names + interaction_names
        
        y = np.argmax(best_u, axis=0)
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='saga')
        
        ext_score = cross_val_score(clf, X_extended, y, cv=5, scoring='accuracy').mean()
        logger.info(f"{method_name} - Extended feature set classification accuracy: {ext_score:.4f}")
        print(f"{method_name} - Extended feature set classification accuracy: {ext_score:.4f}")
        
        clf.fit(X_extended, y)
        feature_importance = np.abs(clf.coef_).sum(axis=0)
        
        top_indices = np.argsort(feature_importance)[-5:]
        top_interaction_features = [(extended_feature_names[idx], feature_importance[idx]) for idx in top_indices[::-1]]
        
        logger.info(f"{method_name} - Important features in extended set:")
        for name, importance in top_interaction_features:
            logger.info(f"  {name}: {importance:.4f}")
            print(f"  {name}: {importance:.4f}")
    
    # Detailed feature scores
    logger.info(f"{method_name} - Feature score details:")
    for j in sorted(scores, key=scores.get, reverse=True):
        logger.info(f"{feature_names[j]}: {scores[j]:.4f}")
    
    # Return analysis results
    return {
        'top_features': top_features_names,
        'single_feature_ranking': single_feature_ranking,
        'best_pair': best_pair if len(stable_idxs) >= 2 else None,
        'best_pair_score': best_pair_score if len(stable_idxs) >= 2 else None,
        'subset_results': subset_results,
        'extended_score': ext_score if len(stable_idxs) >= 2 else None,
        'top_interactions': top_interaction_features if len(stable_idxs) >= 2 else None
    }

def main():
    start_time = time.time()

    # Initialize ArgumentParser only once
    parser = argparse.ArgumentParser(description='Multi-Method Feature Selection Program')
    # REMOVED: Original dataset-specific paths
    # TODO: Update these paths to point to your dataset location
    parser.add_argument('--audio_dir', type=str, default='./data/audio_clips', 
                       help='Audio directory - UPDATE THIS PATH')
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', 
                       help='Training CSV file - UPDATE THIS PATH')
    # REMOVED: Hardcoded model path
    # TODO: Update this path to your model location
    parser.add_argument('--model_path', type=str, default='./models/data2vec-audio-large-960h', 
                       help='data2vec model path - UPDATE THIS PATH')
    parser.add_argument('--hidden_dim', type=int, default=10, help='Reduced dimensionality for hidden states')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU available')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specify GPU ID to use, if not set uses CUDA_VISIBLE_DEVICES environment variable or auto-select')
    
    parser.add_argument('--k_filter', type=int, default=7, help='Number of features selected by Filter method')
    parser.add_argument('--k_embedded', type=int, default=7, help='Number of features selected by Embedded method (RFE/Lasso)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of feature selection iterations')
    parser.add_argument('--stability_threshold', type=float, default=0.6, help='Feature stability threshold')
    
    parser.add_argument('--method', type=str, default='ALL', choices=['FILTER', 'RFE', 'LASSO', 'FILTER_LASSO', 'ALL'], 
                        help='Feature selection method to use (FILTER, RFE, LASSO, FILTER_LASSO, or ALL for all methods)')
    
    args = parser.parse_args()

    # Device setup (CPU/GPU)
    if args.force_cpu:
        # Force CPU usage
        device = 'cpu'
        print("Force CPU mode enabled")
    else:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Get available GPU count
            gpu_count = torch.cuda.device_count()
            
            if gpu_count == 0:
                device = 'cpu'
                print("No available GPU detected, using CPU")
            else:
                # Try to find available GPUs
                available_gpus = []
                for i in range(gpu_count):
                    try:
                        # Try to perform a simple operation on this GPU to check availability
                        with torch.cuda.device(i):
                            torch.tensor([1.0], device=f'cuda:{i}')
                        available_gpus.append(i)
                        print(f"GPU {i} ({torch.cuda.get_device_name(i)}) available")
                    except Exception as e:
                        print(f"GPU {i} ({torch.cuda.get_device_name(i)}) unavailable: {str(e)}")
                
                if not available_gpus:
                    device = 'cpu'
                    print("All GPUs unavailable, using CPU")
                else:
                    # If GPU ID is specified, use specified GPU
                    if args.gpu_id is not None and args.gpu_id in available_gpus:
                        device = f'cuda:{args.gpu_id}'
                        print(f"Using specified GPU {args.gpu_id}")
                    # Otherwise use the first available GPU
                    else:
                        if args.gpu_id is not None and args.gpu_id not in available_gpus:
                            print(f"Warning: Specified GPU {args.gpu_id} unavailable")
                        
                        # Use the first available GPU found
                        device = f'cuda:{available_gpus[0]}'
                        print(f"Using first available GPU: {available_gpus[0]} ({torch.cuda.get_device_name(available_gpus[0])})")
        else:
            device = 'cpu'
            print("CUDA unavailable, using CPU")
    
    print(f"Final selected device: {device}")

    audio_dir = args.audio_dir
    train_csv = args.train_csv
    model_path = args.model_path
    hidden_dim = args.hidden_dim
    method = args.method.upper()
    
    # REMOVED: Hardcoded speaker list
    # TODO: Update this list based on your dataset
    client_spks = [
        # Add your speaker IDs here
        'SPEAKER_001', 'SPEAKER_002', 'SPEAKER_003'  # EXAMPLE - replace with actual IDs
    ]
    
    basic_features = [
        "voiced_rate","pause_rate","word_rate","f0_std",
        "energy_mean","energy_std","mfcc_mean","pause_mean_length",
        "pause_count","speech_rate_variance"
    ]
    
    C_max           = 10
    fuzziness       = 2.0
    
    K_filter        = args.k_filter
    K_embedded      = args.k_embedded
    M               = args.iterations
    theta           = args.stability_threshold

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV file does not exist: {train_csv}")

    logger = setup_logger(f'feature_selection_{method.lower()}.log')
    logger.info("=== STARTING FEATURE SELECTION ===")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Training CSV: {train_csv}")
    logger.info(f"Basic features: {basic_features}")
    logger.info(f"Will also extract hidden states with dimensionality reduced to {hidden_dim}")
    logger.info(f"Hyperparameters: K_filter={K_filter}, K_embedded={K_embedded}, M={M}, theta={theta}")
    logger.info(f"Selected method: {method}")

    # MODIFIED: Read CSV with flexible format
    try:
        df = pd.read_csv(train_csv, dtype={'sentence': str})
        logger.info(f"Loaded CSV with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        logger.error("Please ensure your CSV has the required columns: 'spk', 'path', 'sentence'")
        return

    # MODIFIED: Filter speakers based on provided list
    all_audio_records = []
    for spk_prefix in client_spks:
        spk_records = df[df['spk'].str.startswith(spk_prefix)].copy()
        if len(spk_records) > 0:
            logger.info(f"Found {len(spk_records)} records for speaker {spk_prefix}")
            all_audio_records.append(spk_records)
        else:
            logger.warning(f"No records found for speaker {spk_prefix}")
    
    if all_audio_records:
        filtered_df = pd.concat(all_audio_records)
        logger.info(f"Total records found: {len(filtered_df)}")
    else:
        logger.error("No records found for specified speakers, program terminated")
        return
    
    audio_files = []
    audio_sr = []
    transcripts = []
    valid_indices = []
    
    logger.info("Collecting audio files and transcripts...")
    for i, (_, row) in enumerate(tqdm(filtered_df.iterrows(), desc="Reading audio files", total=len(filtered_df))):
        if pd.isna(row['sentence']) or str(row['sentence']).strip() == '':
            logger.warning(f"Skipping record {row['path']} because sentence is empty")
            continue
            
        transcript = str(row['sentence']).strip()
        wav_path = os.path.join(audio_dir, row['path'])
        
        if not os.path.exists(wav_path):
            logger.warning(f"Audio file does not exist: {wav_path}")
            continue
        
        try:
            audio, sr = sf.read(wav_path)
            audio_files.append(audio)
            audio_sr.append(sr)
            transcripts.append(transcript)
            valid_indices.append(i)
        except Exception as e:
            logger.error(f"Error reading {wav_path}: {str(e)}")
    
    logger.info(f"Successfully read {len(audio_files)} audio files")
    
    X_basic = []
    logger.info("Extracting basic features...")
    
    for i, (audio, sr, transcript) in enumerate(tqdm(zip(audio_files, audio_sr, transcripts), desc="Extracting basic features", total=len(audio_files))):
        try:
            feat_dict = extract_key_features(
                audio_signal=audio,
                sample_rate=sr,
                transcript=transcript,
                selected_features=basic_features
            )
            
            row_features = []
            for feat in basic_features:
                v = feat_dict.get(feat, 0.0)
                if isinstance(v, (list, np.ndarray)):
                    v = float(np.mean(v))
                row_features.append(float(v))
            
            X_basic.append(row_features)
        except Exception as e:
            logger.error(f"Error extracting features from file {i}: {str(e)}")
            X_basic.append([0.0] * len(basic_features))
    
    X_basic = np.array(X_basic)
    logger.info(f"Basic features matrix shape: {X_basic.shape}")
    
    try:
        clean_gpu_memory()
        X_hidden = extract_and_process_hidden_states(
            audio_files=audio_files,
            transcripts=transcripts,
            sample_rates=audio_sr,
            model_path=model_path,
            device=device,
            n_components=hidden_dim,
            logger=logger
        )
        
        if X_hidden is None:
            logger.warning("Hidden state extraction failed, using only basic features")
            X = X_basic
            feature_names = basic_features
        else:
            logger.info(f"Hidden states matrix (after reduction) shape: {X_hidden.shape}")
            
            X = np.hstack([X_basic, X_hidden])
            
            hidden_feature_names = [f"hidden_comp_{i+1}" for i in range(hidden_dim)]
            feature_names = basic_features + hidden_feature_names
            
            logger.info(f"Combined feature matrix shape: {X.shape}")
            logger.info(f"All feature names: {feature_names}")
    except Exception as e:
        logger.error(f"Error during hidden state extraction process: {str(e)}")
        logger.warning("Using only basic features")
        X = X_basic
        feature_names = basic_features
    
    clean_gpu_memory()
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    logger.info("Feature standardization completed")

    best_xb = np.inf
    best_c = None
    best_u = None
    best_cntr = None
    
    for c in tqdm(range(2, C_max+1), desc="Finding optimal cluster number"):
        cntr, u, *_ = fuzz.cluster.cmeans(
            X_scaled.T, c, fuzziness,
            error=0.005, maxiter=1000, init=None
        )
        xb = compute_xb(u, cntr, X_scaled, m=fuzziness)
        logger.info(f"  c={c} → XB={xb:.4f}")
        if xb < best_xb:
            best_xb, best_c, best_u, best_cntr = xb, c, u.copy(), cntr.copy()
    
    logger.info(f"Selected cluster number: c*={best_c} (XB={best_xb:.4f})")

    # Store results
    results = {}
    
    # Execute feature selection based on selected method
    if method in ['FILTER', 'ALL']:
        stable_idxs_filter, scores_filter = run_filter_feature_selection(
            X_scaled, best_u, feature_names, K_filter, M, theta, logger
        )
        results['FILTER'] = analyze_features(
            X_scaled, best_u, stable_idxs_filter, scores_filter, feature_names, logger, "FILTER"
        )
    
    if method in ['RFE', 'ALL']:
        stable_idxs_rfe, scores_rfe = run_rfe_feature_selection(
            X_scaled, best_u, feature_names, K_embedded, M, theta, logger
        )
        results['RFE'] = analyze_features(
            X_scaled, best_u, stable_idxs_rfe, scores_rfe, feature_names, logger, "RFE"
        )
    
    if method in ['LASSO', 'ALL']:
        stable_idxs_lasso, scores_lasso = run_lasso_feature_selection(
            X_scaled, best_u, feature_names, K_embedded, M, theta, logger
        )
        results['LASSO'] = analyze_features(
            X_scaled, best_u, stable_idxs_lasso, scores_lasso, feature_names, logger, "LASSO"
        )
    
    if method in ['FILTER_LASSO', 'ALL']:
        stable_idxs_fl, scores_fl = run_filter_lasso_feature_selection(
            X_scaled, best_u, feature_names, K_filter, K_embedded, M, theta, logger
        )
        results['FILTER_LASSO'] = analyze_features(
            X_scaled, best_u, stable_idxs_fl, scores_fl, feature_names, logger, "FILTER_LASSO"
        )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info("\n=== FEATURE SELECTION SUMMARY ===")
    logger.info(f"Program execution time: {str(timedelta(seconds=int(execution_time)))}")
    
    # Print comparison results
    if method == 'ALL':
        logger.info("\n=== COMPARISON OF METHODS ===")
        print("\n=== COMPARISON OF METHODS ===")
        
        # Compare top features
        logger.info("Top features by method:")
        print("Top features by method:")
        
        for method_name, result in results.items():
            if result and 'top_features' in result:
                logger.info(f"  {method_name}: {result['top_features']}")
                print(f"  {method_name}: {result['top_features']}")
        
        # Compare best feature pairs
        logger.info("\nBest feature pairs by method:")
        print("\nBest feature pairs by method:")
        
        for method_name, result in results.items():
            if result and 'best_pair' in result and result['best_pair']:
                logger.info(f"  {method_name}: {result['best_pair']} (score: {result['best_pair_score']:.4f})")
                print(f"  {method_name}: {result['best_pair']} (score: {result['best_pair_score']:.4f})")
        
        # Compare best subsets
        logger.info("\nBest feature subsets by method:")
        print("\nBest feature subsets by method:")
        
        for method_name, result in results.items():
            if result and 'subset_results' in result:
                for size, (subset, score) in result['subset_results'].items():
                    logger.info(f"  {method_name} (size {size}): {subset} (score: {score:.4f})")
                    print(f"  {method_name} (size {size}): {subset} (score: {score:.4f})")
    
    logger.info("\n=== FEATURE SELECTION COMPLETE ===")
    print(f"\n=== FEATURE SELECTION COMPLETE ===")
    print(f"Program execution time: {str(timedelta(seconds=int(execution_time)))}")
    print(f"See {logger.handlers[0].baseFilename} for full results")
    print(f"Note: To specify GPU, use: CUDA_VISIBLE_DEVICES=X python feature_selection_multi.py [args]")

if __name__ == '__main__':
    main()