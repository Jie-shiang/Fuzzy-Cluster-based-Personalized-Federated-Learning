#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from transformers import Wav2Vec2Processor
import pickle
from sklearn.cluster import KMeans
from transformers import Data2VecAudioConfig
from collections import Counter
import torch
import shutil
import copy
import datetime

# Import from custom library
from options import args_parser
# REMOVED: Direct import with hardcoded paths
# TODO: Update the utils import based on your project structure
from utils import get_raw_dataset, exp_details, add_cluster_id, load_model, evaluateASR, average_weights
from update_CPFL import update_network_weight, map_to_result, map_to_result_MMSE
from training import client_train, centralized_training, client_getEmb

# Create experiment log directory
os.makedirs('./logs_exp', exist_ok=True)

# Add simplified logging function
def exp_log(message, print_also=True):
    """
    Record key messages to experiment log
    
    Args:
        message: Message to record
        print_also: Whether to print to console
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    log_path = os.path.join('./logs_exp', 'experiment_log.txt')
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    
    if print_also:
        print(message)

def FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, init_global_weights=None):
    exp_log(f"\n===== Starting global training round {epoch+1} =====")
    exp_log(f"Model input path: {model_in_path_root}")
    exp_log(f"Model output path: {model_out_path}")
    
    global_weights_lst = []
    
    for cluster_id in tqdm(range(args.num_lms)):
        model_id = cluster_id
        
        exp_log(f"\n----- Cluster {cluster_id} training starts -----")
        
        if args.num_lms != 1:
            print(f'\n | Global Training Round for Cluster {cluster_id}: {epoch+1} |\n')
        else:
            print(f'\n | Global Training Round: {epoch+1} |\n')
            cluster_id = None

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        exp_log(f"Selected client IDs: {idxs_users}")
        
        local_weights_en = []
        local_weights_de = []
        num_training_samples_lst = []
        membership_weights_lst = []
        
        client_log_path = f"client_details_round{epoch+1}_cluster{cluster_id}.txt"
        
        for idx in idxs_users:
            exp_log(f"Starting training client {idx}")
            
            if init_global_weights is not None:
                global_weights = init_global_weights[model_id]
                exp_log(f"Using initialized global weights")
            else:
                global_weights = None
                exp_log(f"Not using initialized global weights")

            if args.use_soft_clustering:
                client_dataset = train_dataset_supervised.filter(
                    lambda x: 'valid_clusters' in x and cluster_id in x['valid_clusters']
                )
                exp_log(f"Client {idx} valid samples in cluster {cluster_id}: {len(client_dataset)}")
                
                if len(client_dataset) == 0:
                    exp_log(f"Client {idx} has no valid samples in cluster {cluster_id}, skipping...")
                    continue
                
                if args.use_membership_weighted_avg:
                    avg_membership = np.mean([
                        example['normalized_memberships'][cluster_id] 
                        for example in client_dataset
                    ])
                    membership_weights_lst.append(avg_membership)
                    exp_log(f"Client {idx} average membership weight in cluster {cluster_id}: {avg_membership}")
            
            if epoch > 0:
                clean_model_in_path_root = model_in_path_root.replace("_FLASR", "")
                client_model_path = clean_model_in_path_root + "_client" + str(idx) + "_round" + str(epoch-1)
                if cluster_id != None:
                    client_model_path += "_cluster" + str(cluster_id)
                client_model_path += "_TrainingAddress/final/"
                
                path_exists = os.path.exists(client_model_path)
                exp_log(f"Checking client model path: {client_model_path}")
                exp_log(f"Path exists: {path_exists}")
                
                if not path_exists:
                    exp_log(f"Warning: Client model path does not exist! Will attempt training, may fail.")
            
            try:
                final_result = client_train(
                    args, model_in_path_root, model_out_path, 
                    train_dataset_supervised, test_dataset, 
                    idx, epoch, cluster_id, global_weights
                )
                
                w, num_training_samples = final_result
                
                exp_log(f"Client {idx} training completed, training samples: {num_training_samples}")
                
                if num_training_samples > 0:
                    local_weights_en.append(copy.deepcopy(w[0]))
                    local_weights_de.append(copy.deepcopy(w[1]))
                    num_training_samples_lst.append(num_training_samples)
                    exp_log(f"Client {idx} weights added to aggregation list")
                else:
                    exp_log(f"Client {idx} has no training samples, weights not added")
            
            except Exception as e:
                exp_log(f"Client {idx} training failed: {str(e)}")
                continue
        
        exp_log(f"Number of clients participating in cluster {cluster_id}: {len(local_weights_en)}")
        
        if len(local_weights_en) == 0:
            exp_log(f"No clients participated in cluster {cluster_id}, using previous weights")
            if init_global_weights is not None:
                global_weights_lst.append(init_global_weights[model_id])
                exp_log(f"Using initialized weights")
            else:
                global_weights = [None, None]
                global_weights_lst.append(global_weights)
                exp_log(f"No available weights")
            continue
        
        exp_log(f"Aggregating client weights")
        membership_weights = membership_weights_lst if args.use_membership_weighted_avg else None
        global_weights = [
            average_weights(local_weights_en, num_training_samples_lst, 
                          membership_weights, args),
            average_weights(local_weights_de, num_training_samples_lst, 
                          membership_weights, args)
        ]
        global_weights_lst.append(global_weights)
        exp_log(f"Cluster {cluster_id} weight aggregation completed")
    
    exp_log(f"===== Global training round {epoch+1} completed =====\n")
    return global_weights_lst

# Train 1 round for all clusters at once
def CPFL_training_clusters(args, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, Nth_Kmeans_update=None, init_weights_lst=None):
    """
    Execute multi-round clustering-based training
    Args:
        args: Parameter settings
        model_in_path_root: Model input path
        model_out_path: Model output path
        train_dataset_supervised: Training dataset
        test_dataset: Test dataset
        Nth_Kmeans_update: Whether to update clustering (None means no update)
        init_weights_lst: Initialization weight list
    Returns:
        global_weights_lst: Final model weight list
    """
    exp_log("\n===== CPFL training starts =====")
    exp_log(f"Model input path: {model_in_path_root}")
    exp_log(f"Model output path: {model_out_path}")
    
    if Nth_Kmeans_update == None:
        epochs = args.epochs
        exp_log(f"Training rounds: {epochs} (no clustering update needed)")
    else:
        epochs = args.N_Kmeans_update
        exp_log(f"Training rounds: {epochs} (clustering update #{Nth_Kmeans_update})")
    
    global_weights_lst = init_weights_lst
    exp_log(f"Global weights list initialization status: {'Initialized' if init_weights_lst is not None else 'Not initialized'}")
    
    for i in range(epochs):
        if Nth_Kmeans_update == None:
            epoch = i
        else:
            epoch = int(i + Nth_Kmeans_update*args.N_Kmeans_update)
        
        exp_log(f"\n----- Starting round {epoch+1}/{args.epochs} -----")
        
        # Check source model path
        source_path = args.model_out_path+"_FLASR_global/final/"
        if not os.path.exists(source_path):
            exp_log(f"Warning: Source model path does not exist: {source_path}")
        
        global_weights_lst = FL_training_clusters_loop(args, epoch, model_in_path_root, model_out_path, train_dataset_supervised, test_dataset, global_weights_lst)
        
        exp_log(f"Saving cluster models")
        
        for j in range(args.num_lms):
            # Check if cluster has valid training results for soft clustering
            if args.use_soft_clustering:
                if global_weights_lst[j][0] is None:  # Check encoder weights
                    exp_log(f"Skipping model save for cluster {j}, no valid training occurred")
                    continue
            
            global_weights = global_weights_lst[j]
            
            folder_to_save = args.model_out_path+"_cluster" + str(j) + "_CPFLASR_global_round" + str(epoch)
            exp_log(f"Saving cluster {j} model to: {folder_to_save}")
            
            # Ensure directory exists
            os.makedirs(folder_to_save + "/final", exist_ok=True)
            
            try:
                # Update and save model weights
                model = update_network_weight(args=args, source_path=source_path, target_weight=global_weights, network="ASR")
                model.save_pretrained(folder_to_save + "/final")
                exp_log(f"Cluster {j} model saved successfully")
            except Exception as e:
                exp_log(f"Cluster {j} model save failed: {str(e)}")
            
        model_in_path = args.model_in_path
        exp_log(f"Evaluating model")
        try:
            evaluateASR(args, epoch, test_dataset, train_dataset_supervised)
            exp_log(f"{args.model_out_path}#_CPFLASR_global_round{epoch} evaluation completed.")
        except Exception as e:
            exp_log(f"Model evaluation failed: {str(e)}")
        
        for j in range(args.num_lms):
            temp_folder = args.model_out_path+"_cluster" + str(j) + "_CPFLASR_global_round" + str(epoch)
            if os.path.exists(temp_folder):
                exp_log(f"Cleaning temporary folder: {temp_folder}")
                shutil.rmtree(temp_folder)
            
        args.model_in_path = model_in_path
        torch.cuda.empty_cache()
        exp_log(f"Round {epoch+1} completed, CUDA memory cleared")
    
    exp_log("===== CPFL training completed =====\n")
    return global_weights_lst

class FuzzyCMeansWrapper:
    """
    Wrapper for Fuzzy C-means clustering model
    """
    def __init__(self, cntr, u_matrix, threshold):
        self.cluster_centers_ = cntr
        self.u_matrix = u_matrix.T  # [n_samples, n_clusters]
        self.threshold = threshold

    def predict(self, X, return_full=False):
        """
        Predict cluster memberships for new data
        Args:
            X: Input data
            return_full: Whether to return full membership matrix
        Returns:
            if return_full: membership matrix [n_samples, n_clusters]
            else: cluster assignments [n_samples]
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = np.array(X).T
        
        try:
            import skfuzzy as fuzz
            # Get fuzzy_m parameter
            from options import args_parser
            args = args_parser()
            
            u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X, 
                self.cluster_centers_,
                args.fuzzy_m,
                error=0.005,
                maxiter=1000,
            )
            memberships = u.T
            
            if return_full:
                return memberships
            else:
                return np.argmax(memberships, axis=1)
        except Exception as e:
            exp_log(f"FCM prediction failed: {str(e)}")
            if return_full:
                return np.random.rand(X.shape[0], len(self.cluster_centers_))
            else:
                return np.random.randint(0, len(self.cluster_centers_), size=X.shape[0])

    def __getstate__(self):
        """
        For serialization support
        """
        return {
            'cluster_centers_': self.cluster_centers_,
            'u_matrix': self.u_matrix,
            'threshold': self.threshold
        }

    def __setstate__(self, state):
        """
        For serialization support
        """
        self.cluster_centers_ = state['cluster_centers_']
        self.u_matrix = state['u_matrix']
        self.threshold = state['threshold']

def build_clustering_model(args, dx):
    """
    Build clustering model, supports K-means and Fuzzy C-means
    Args:
        args: Parameter settings
        dx: Input features
    Returns:
        clustering model (KMeans or FuzzyCMeansWrapper)
    """
    exp_log("\n===== Building clustering model =====")
    
    # Ensure data format is correct - convert to numpy array
    dx = np.array(dx)
    
    if args.use_soft_clustering:
        exp_log("Using Fuzzy C-means clustering...")
        try:
            import skfuzzy as fuzz
        except ImportError:
            exp_log("Please install skfuzzy: pip install scikit-fuzzy")
            return None
            
        # Ensure data is 2D
        if len(dx.shape) == 1:
            dx = dx.reshape(-1, 1)
            
        # Transpose data to meet skfuzzy requirements [features, samples]
        dx_fcm = dx.T
        
        # Execute FCM
        try:
            cntr, u_matrix, _, _, _, _, _ = fuzz.cluster.cmeans(
                dx_fcm, 
                c=args.num_lms,
                m=args.fuzzy_m,
                error=0.005,
                maxiter=1000,
                init=None
            )
            
            # Create wrapper instance
            kmeans = FuzzyCMeansWrapper(cntr, u_matrix, args.membership_threshold)
            exp_log("FCM clustering completed successfully")
            
            # Save clustering results
            with open(args.Kmeans_model_path, 'wb') as f:
                pickle.dump(kmeans, f)
            
            return kmeans
            
        except Exception as e:
            exp_log(f"FCM clustering failed: {str(e)}")
            exp_log("Falling back to K-means...")
            return build_kmeans_model(args, dx)
    else:
        exp_log("Using traditional K-means...")
        return build_kmeans_model(args, dx)
    
def build_kmeans_model(args, dx):
    """
    Modified K-means implementation ensuring correct data format
    """
    # Ensure data is numpy array
    dx = np.array(dx)
    
    # Ensure data is 2D
    if len(dx.shape) == 1:
        dx = dx.reshape(-1, 1)
    
    exp_log(f"Input feature shape: {dx.shape}")
    
    kmeans = KMeans(n_clusters=args.num_lms)
    kmeans.fit(dx)
    exp_log("K-means clustering completed")
    
    # Save model
    with open(args.Kmeans_model_path, 'wb') as f:
        pickle.dump(kmeans, f)

    # Get clustering results
    cluster_id_lst = kmeans.predict(dx).tolist()
    counter = Counter(cluster_id_lst)
    result = [counter[i] for i in range(args.num_lms)]
    exp_log(f"Overall cluster sample counts: {result}")

    # Record results
    path = './logs/Kmeans_log.txt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write("---------------------------------------------------\n")
        f.write("Cluster centers: " + str(kmeans.cluster_centers_) + "\n")
        f.write("Overall cluster sample counts: " + str(result) + "\n")
        f.write("---------------------------------------------------\n")

    return kmeans

def get_clients_representations(args, model_in_path, train_dataset_supervised, test_dataset, TEST, cluster_id=None):
    """
    Modified feature extraction function ensuring correct output format
    """
    exp_log("\n===== Starting client feature extraction =====")
    
    multiprocessing.set_start_method('spawn', force=True)

    idxs_users = np.random.choice(range(args.num_users), args.num_users, replace=False)
    pool = multiprocessing.Pool(processes=args.num_users)

    try:
        final_result = pool.starmap_async(
            client_getEmb, 
            [(args, model_in_path, train_dataset_supervised, test_dataset, idx, cluster_id, TEST) 
             for idx in idxs_users]
        )
        final_result.wait()
        results = final_result.get()
    except Exception as e:
        exp_log(f"Feature extraction error: {str(e)}")
        raise
    finally:
        pool.close()
        pool.join()

    # Initialize lists
    hidden_states_mean_lst = []
    loss_lst = []
    entropy_lst = []
    vocab_ratio_rank_lst = []
    encoder_attention_1D_lst = []
    speech_features_lst = []
    
    # Integrate all features
    for result in results:
        if result is not None:
            hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features = result
            hidden_states_mean_lst.extend(hidden_states_mean)
            loss_lst.extend(loss)
            entropy_lst.extend(entropy)
            vocab_ratio_rank_lst.extend(vocab_ratio_rank)
            encoder_attention_1D_lst.extend(encoder_attention_1D)
            speech_features_lst.extend(speech_features)
    
    exp_log(f"Feature extraction completed. Sample count: {len(speech_features_lst)}")
    
    return [
        np.array(hidden_states_mean_lst),
        np.array(loss_lst),
        np.array(entropy_lst),
        np.array(vocab_ratio_rank_lst),
        np.array(encoder_attention_1D_lst),
        np.array(speech_features_lst)
    ]

# REMOVED: Environment variable for dataset path
# TODO: Update this based on your dataset location
# CPFL_dataRoot = os.environ.get('CPFL_dataRoot')
CPFL_dataRoot = './data'  # UPDATE THIS PATH

def assign_cluster(args, dataset, kmeans, dataset_path, csv_path):
    """
    Assign clustering results to samples in dataset and save detailed statistics
    Args:
        args: Parameter settings
        dataset: Dataset to assign
        kmeans: Clustering model (KMeans or FuzzyCMeansWrapper)
        dataset_path: Dataset save path
        csv_path: Original csv path
    Returns:
        Dataset with added cluster information
    """
    exp_log("\n===== Starting cluster assignment =====")
    exp_log(f"Dataset size: {len(dataset)}")
    
    torch.set_num_threads(1)
    
    # Load ASR model
    mask_time_prob = 0
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
    model = load_model(args, args.model_in_path, config)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

    # Get features
    _, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features = map_to_result_MMSE(dataset[0], processor, model, 0)

    for i in range(len(dataset) - 1):
        _, hidden_states_mean_2, loss2, entropy2, vocab_ratio_rank2, encoder_attention_1D2, speech_features2 = map_to_result_MMSE(dataset[i+1], processor, model, i+1)
        
        hidden_states_mean.extend(hidden_states_mean_2)
        loss.extend(loss2)
        entropy.extend(entropy2)
        vocab_ratio_rank.extend(vocab_ratio_rank2)
        encoder_attention_1D.extend(encoder_attention_1D2)
        speech_features.extend(speech_features2)
        print("\r"+ str(i), end="")

    # Prepare feature data
    speech_features_2d = np.array(speech_features).reshape(-1, 1) if len(np.array(speech_features).shape) == 1 else np.array(speech_features)
    
    # Statistics dictionary
    stats = {
        'sample_id': [],
        'speaker_id': [],  # Add speaker_id
        'cluster_id': [],
        'cluster_memberships': [],
        'valid_clusters': [],
        'normalized_memberships': [],
        'labels': []  # MODIFIED: Use generic 'labels' instead of 'dementia_labels'
    }
    
    if args.use_soft_clustering:
        exp_log("\nUsing Fuzzy C-means assignment...")
        try:
            memberships = kmeans.predict(speech_features_2d, return_full=True)
            cluster_id_lst = memberships.tolist()
            
            def add_cluster_memberships(example, memberships):
                """Add FCM cluster information to sample"""
                stats['sample_id'].append(example['path'])
                stats['speaker_id'].append(example['path'].split('_')[0])  # Extract speaker_id
                # MODIFIED: Use generic label key
                stats['labels'].append(example.get('labels', example.get('dementia_labels', 0)))
                
                example['cluster_memberships'] = memberships
                stats['cluster_memberships'].append(memberships)
                
                example['cluster_id'] = int(np.argmax(memberships))
                stats['cluster_id'].append(example['cluster_id'])
                
                valid_clusters = np.where(np.array(memberships) >= args.membership_threshold)[0]
                example['valid_clusters'] = valid_clusters.tolist()
                stats['valid_clusters'].append(example['valid_clusters'])
                
                valid_sum = sum(memberships[i] for i in valid_clusters)
                
                if len(valid_clusters) > 0:
                    normalized_memberships = [
                        memberships[i]/valid_sum if i in valid_clusters else 0.0 
                        for i in range(len(memberships))
                    ]
                else:
                    max_idx = np.argmax(memberships)
                    normalized_memberships = [
                        1.0 if i == max_idx else 0.0 
                        for i in range(len(memberships))
                    ]
                    
                example['normalized_memberships'] = normalized_memberships
                stats['normalized_memberships'].append(normalized_memberships)
                
                return example
                
            dataset = dataset.map(
                lambda example: add_cluster_memberships(example, cluster_id_lst.pop(0))
            )
            
            # Save clustering statistics to CSV
            import pandas as pd
            import os
            
            # Convert to DataFrame
            df = pd.DataFrame(stats)
            
            # Ensure results directory exists
            results_dir = './results'  # MODIFIED: Use relative path
            os.makedirs(results_dir, exist_ok=True)
            
            # Save basic statistics
            stats_path = os.path.join(results_dir, f'clustering_stats_round{args.FL_STAGE}.csv')
            df.to_csv(stats_path, index=False)
            exp_log(f"\nClustering statistics saved to: {stats_path}")
            
            # Calculate and save detailed statistics
            detailed_stats = {
                'cluster_id': [],
                'total_samples': [],
                'primary_samples': [],
                'valid_samples': [],
                'avg_membership': [],
                'avg_normalized_membership': [],
                'class_0_samples': [],  # MODIFIED: Generic class names
                'class_1_samples': []   # MODIFIED: Generic class names
            }
            
            for c in range(args.num_lms):
                detailed_stats['cluster_id'].append(c)
                cluster_samples = df[df['cluster_id'] == c]
                
                detailed_stats['total_samples'].append(len(df))
                detailed_stats['primary_samples'].append(len(cluster_samples))
                detailed_stats['valid_samples'].append(sum(1 for x in df['valid_clusters'] if c in x))
                
                avg_membership = np.mean([x[c] for x in df['cluster_memberships']])
                detailed_stats['avg_membership'].append(avg_membership)
                
                avg_norm_membership = np.mean([x[c] for x in df['normalized_memberships']])
                detailed_stats['avg_normalized_membership'].append(avg_norm_membership)
                
                # Calculate class distribution
                class_0_samples = len(cluster_samples[cluster_samples['labels'] == 0])
                class_1_samples = len(cluster_samples[cluster_samples['labels'] == 1])
                detailed_stats['class_0_samples'].append(class_0_samples)
                detailed_stats['class_1_samples'].append(class_1_samples)
            
            # Save detailed statistics
            detailed_stats_path = os.path.join(results_dir, f'detailed_clustering_stats_round{args.FL_STAGE}.csv')
            pd.DataFrame(detailed_stats).to_csv(detailed_stats_path, index=False)
            
            # Print main statistics
            n_samples = len(dataset)
            valid_clusters_count = [len(example['valid_clusters']) for example in dataset]
            exp_log(f"\nAssignment statistics: Total samples: {n_samples}, Average valid clusters per sample: {np.mean(valid_clusters_count):.2f}")
            
        except Exception as e:
            exp_log(f"\nFCM assignment failed: {str(e)}")
            exp_log("Falling back to hard assignment...")
            memberships = kmeans.predict(speech_features_2d, return_full=False)
            cluster_id_lst = memberships.tolist()
            dataset = dataset.map(
                lambda example: add_cluster_id(example, cluster_id_lst.pop(0))
            )
    else:
        exp_log("\nUsing K-means assignment...")
        cluster_id_lst = kmeans.predict(speech_features_2d).tolist()
        dataset = dataset.map(
            lambda example: add_cluster_id(example, cluster_id_lst.pop(0))
        )

    # Save dataset
    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    dataset.save_to_disk(stored+"_temp")
    if os.path.exists(stored):
        shutil.rmtree(stored)
    os.rename(stored+"_temp", stored)
    
    exp_log(f"\nDataset with cluster information saved, path: {stored}")
    return dataset

# FL stage 1: Global train ASR encoder & decoder
def GlobalTrainASR(args, train_dataset_supervised, test_dataset):
    args.local_ep = args.global_ep
    args.STAGE = 0
    
    # Add MixSpeech related logs
    if args.use_mixspeech:
        exp_log(f"Applying MixSpeech, parameters: alpha={args.mixspeech_alpha}, max_lambda={args.mixspeech_max_lambda}, prob={args.mixspeech_prob}")
    
    centralized_training(args=args, model_in_path=args.pretrain_name, model_out_path=args.model_out_path+"_finetune", 
                        train_dataset=train_dataset_supervised, test_dataset=test_dataset, epoch=0)          

# FL stage 3: perform k-means clustering
def Kmeans_clustering(args, train_dataset_supervised, test_dataset):
    """
    Stage 3: Execute clustering (K-means or Fuzzy C-means)
    Args:
        args: Parameter settings
        train_dataset_supervised: Training dataset
        test_dataset: Test dataset
    """
    exp_log("\n===== Starting Stage 3 - Clustering =====")
    
    # 1. Get feature representations
    try:
        hidden_states_mean_lst, loss_lst, entropy_lst, vocab_ratio_rank_lst, encoder_attention_1D_lst, speech_features_lst = get_clients_representations(
            args=args, 
            model_in_path=args.model_in_path, 
            train_dataset_supervised=train_dataset_supervised,
            test_dataset=test_dataset, 
            TEST=False, 
            cluster_id=None
        )
        
        # 2. Feature dimension check and report
        exp_log(f"Feature dimensions: entropy={np.shape(np.array(entropy_lst))}, hidden_states={np.shape(np.array(hidden_states_mean_lst))}, speech_features={np.shape(np.array(speech_features_lst))}")
    
        # 3. Execute clustering
        try:
            kmeans = build_clustering_model(args, speech_features_lst)
            if kmeans is None:
                raise Exception("Clustering model creation failed")
        except Exception as e:
            exp_log(f"Clustering failed: {str(e)}")
            return
        
        # 4. Save clustering results to dataset
        exp_log("\nAssigning clusters to dataset...")
        # MODIFIED: Use relative path
        dataset_path = "./dataset/clustered/"
        
        # Ensure save directory exists
        os.makedirs(dataset_path, exist_ok=True)
        
        # Process test set
        exp_log("\nProcessing test dataset...")
        test_dataset = assign_cluster(args, test_dataset, kmeans, dataset_path, csv_path=f"{CPFL_dataRoot}/test.csv")
        
        # Process training set
        exp_log("\nProcessing training dataset...")
        train_dataset = assign_cluster(args, train_dataset_supervised, kmeans, dataset_path, csv_path=f"{CPFL_dataRoot}/train.csv")
        
        exp_log("\nStage 3 - Clustering completed!")
    except Exception as e:
        exp_log(f"Error occurred during Stage 3 clustering process: {str(e)}")
        raise

# FL stage 4: FL train ASR
def CPFL_TrainASR(args, train_dataset_supervised, test_dataset):
    """
    Stage 4: FL training ASR, supports soft clustering
    Args:
        args: Parameter settings
        train_dataset_supervised: Training dataset
        test_dataset: Test dataset
    """
    # Initialize
    global_weights_lst = None
    total_rounds = int(args.epochs / args.N_Kmeans_update)
    
    exp_log("\n===== Starting Stage 4 - FL ASR Training =====")
    exp_log(f"Total rounds: {total_rounds}")
    exp_log(f"Clustering mode: {'Fuzzy C-means' if args.use_soft_clustering else 'K-means'}")
    
    # Check source model path
    source_path = f"{args.model_out_path}_FLASR_global/final/"
    if not os.path.exists(source_path):
        exp_log(f"Warning: Source model path does not exist: {source_path}")
    
    # For each training round
    for i in range(total_rounds):
        exp_log(f"\n===== Starting round {i+1}/{total_rounds} =====")
        
        # Determine if clustering update is needed
        if total_rounds == 1:
            Nth_Kmeans_update = None  # No clustering update needed
        else:
            Nth_Kmeans_update = i
        
        try:
            # Execute one round of training
            global_weights_lst = CPFL_training_clusters(
                args=args,
                model_in_path_root=args.model_in_path+"_FLASR",
                model_out_path=args.model_out_path,
                train_dataset_supervised=train_dataset_supervised,
                test_dataset=test_dataset,
                Nth_Kmeans_update=Nth_Kmeans_update,
                init_weights_lst=global_weights_lst
            )
            
            # Update global model for each cluster
            for j in range(args.num_lms):
                # Check if cluster has valid training results
                if args.use_soft_clustering and global_weights_lst[j][0] is None:
                    exp_log(f"Skipping global model update for cluster {j} - no valid training data")
                    continue
                
                # Get and save model
                global_weights = global_weights_lst[j]
                round_num = int((i+1)*args.N_Kmeans_update - 1)
                folder_to_save = f"{args.model_out_path}_cluster{j}_CPFLASR_global_round{round_num}"
                
                # Ensure directory exists
                os.makedirs(f"{folder_to_save}/final", exist_ok=True)
                
                try:
                    # Update model weights
                    model = update_network_weight(
                        args=args,
                        source_path=source_path,
                        target_weight=global_weights,
                        network="ASR"
                    )
                    model.save_pretrained(f"{folder_to_save}/final")
                except Exception as e:
                    exp_log(f"Model save failed for cluster {j}: {str(e)}")
            
            exp_log(f"Round {i+1} completed - all valid cluster models updated")
            
        except Exception as e:
            exp_log(f"Error in round {i+1}: {str(e)}")
            exp_log("Attempting to continue to next round...")
            continue
        
        # Cleanup after each round
        torch.cuda.empty_cache()
    
    exp_log("\nStage 4 - FL ASR training completed!")

    # Save final model weights
    try:
        final_weights_path = f"{args.model_out_path}_final_weights.pkl"
        with open(final_weights_path, 'wb') as f:
            pickle.dump(global_weights_lst, f)
        exp_log(f"Final weights saved to {final_weights_path}")
    except Exception as e:
        exp_log(f"Error saving final weights: {str(e)}")

def get_dataset(args):                                                                      
    # return train_dataset_supervised, test_dataset
    # MODIFIED: Use generic dataset identifier
    args.dataset = "pathological_speech"  # MODIFIED: Generic name instead of "adress"
    train_dataset_supervised, test_dataset = get_raw_dataset(args)                          # get dataset

    return train_dataset_supervised, test_dataset

if __name__ == '__main__':
    start_time = time.time()                                                                # record starting time

    # Create experiment log directory and initialize logging
    os.makedirs('./logs_exp', exist_ok=True)
    
    # Clear main experiment log file
    with open('./logs_exp/experiment_log.txt', 'w') as f:
        f.write(f"===== Experiment started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n\n")
    
    # Record program start
    exp_log("===== Program started =====")

    path_project = os.path.abspath('..')                                                    # define paths
    
    args = args_parser()                                                                    # get configuration
    
    # Record key configurations
    exp_log("\nKey configuration parameters:")
    exp_log(f"FL_STAGE: {args.FL_STAGE}, num_lms: {args.num_lms}, epochs: {args.epochs}")
    exp_log(f"model_in_path: {args.model_in_path}")
    exp_log(f"model_out_path: {args.model_out_path}")
    
    # Check important paths
    for path in [args.model_in_path, f"{args.model_out_path}_FLASR_global/final/"]:
        path_exists = os.path.exists(path)
        if not path_exists and "FLASR_global" in path:  # Only check FLASR_global in Stage 4
            exp_log(f"Warning: Important path does not exist: {path}")
    
    exp_details(args)                                                                       # print out details based on configuration
    
    try:
        train_dataset_supervised, test_dataset = get_dataset(args)
        exp_log(f"Dataset loaded successfully, training set size: {len(train_dataset_supervised)}, test set size: {len(test_dataset)}")
    except Exception as e:
        exp_log(f"Dataset loading failed: {str(e)}")
        raise

    # Training
    if args.FL_STAGE == 1:                                                                  # train Fine-tuned ASR W_0^G
        exp_log("| Starting FL training stage 1 -- Global ASR training |")
        args.STAGE = 0                                                                      # 0: train ASR encoder & decoder
        try:
            GlobalTrainASR(args=args, train_dataset_supervised=train_dataset_supervised, test_dataset=test_dataset)                      
            exp_log("| FL training stage 1 completed |")
        except Exception as e:
            exp_log(f"FL training stage 1 failed: {str(e)}")
    elif args.FL_STAGE == 2:
        exp_log("| Stage 2 deprecated!! |")
    elif args.FL_STAGE == 3:                                                                # K-means clustering
        exp_log("| Starting FL training stage 3 -- Clustering |")
        try:
            Kmeans_clustering(args, train_dataset_supervised, test_dataset)
            exp_log("| FL training stage 3 completed |")
        except Exception as e:
            exp_log(f"FL training stage 3 failed: {str(e)}")
    elif args.FL_STAGE == 4:                                                                # FL train ASR decoder
        exp_log("| Starting FL training stage 4 -- FL ASR training |")
        args.STAGE = 0                                                                      # train ASR encoder as well
        try:
            CPFL_TrainASR(args, train_dataset_supervised, test_dataset)
            exp_log("| FL training stage 4 completed |")
        except Exception as e:
            exp_log(f"FL training stage 4 failed: {str(e)}")
    else:
        exp_log(f"Only FL training stages 1-4 are available, current FL_STAGE = {args.FL_STAGE}")
    
    end_time = time.time()
    run_time = end_time - start_time
    exp_log(f'\nTotal runtime: {run_time:.4f} seconds ({run_time/60:.2f} minutes)')
    exp_log(f"===== Program ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    
    print('\nTotal runtime: {0:0.4f} seconds'.format(time.time()-start_time))