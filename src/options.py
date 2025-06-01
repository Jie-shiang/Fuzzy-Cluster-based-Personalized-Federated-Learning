#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training, R")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: C")
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs")
    parser.add_argument('--train_batch_size', type=int, default=3, help="")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="")
    # model arguments
    parser.add_argument('--model', type=str, default='data2vec', help='model name')
    # other arguments
    parser.add_argument('--dataset', type=str, default='pathological_speech', help="name of dataset")  # MODIFIED: Generic name
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    # additional arguments
    parser.add_argument('--pretrain_name', type=str, default='facebook/data2vec-audio-large-960h', help="str used to load pretrain model")
    parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
    parser.add_argument('-fl_st', '--FL_STAGE', type=int, default=1, help="Current FL training stage")
    # MODIFIED: Use relative paths instead of absolute paths
    parser.add_argument('-model_in', '--model_in_path', type=str, default="./models/data2vec-audio-large-960h", help="Where the global model is saved - UPDATE THIS PATH")
    parser.add_argument('-model_out', '--model_out_path', type=str, default="./models/output/data2vec-audio-large-960h", help="Where to save the model - UPDATE THIS PATH")
    parser.add_argument('-csv', '--csv_path', type=str, default="data2vec-audio-large-960h", help="name for the csv file")
    # 2023/04/20
    parser.add_argument('-EXTRACT', '--EXTRACT', action='store_true', default=False, help="True: extract embs")
    parser.add_argument('-client_id', '--client_id', type=str, default="public", help="client_id: public, 0, 1 ...")
    # 2023/04/24
    parser.add_argument('--global_ep', type=int, default=30, help="number of epoch for global model")
    parser.add_argument('--GPU_batchsize', type=str, default=None, help="use GPU when cpu is full")
    # 2023/05/18
    parser.add_argument('--num_lms', type=int, default=5, help="number of clusters: K 7")
    # 2023/05/20
    parser.add_argument('--eval_steps', type=int, default=20000, help="")
    # 2023/06/16
    parser.add_argument('--training_type', type=int, default=5, help="supervised(1) / semi-supervised(2) / semi then supervised(3) / supervised then semi (4) / all together(5)")
    # 2023/06/19
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help="learning rate for training ASR")
    # 2023/06/27
    # MODIFIED: Use relative path
    parser.add_argument('-Kmeans_model_path', '--Kmeans_model_path', type=str, default="./models/k_means_model", help="Where to save the K-means model - UPDATE THIS PATH")
    # 2023/07/16
    parser.add_argument('-N_Kmeans_update', '--N_Kmeans_update', type=int, default=5, help="For N rounds, k-means model will re-assign cluster")
    # 2023/07/28
    # MODIFIED: Use relative path
    parser.add_argument('-dataset_path_root', '--dataset_path_root', type=str, default="./dataset", help="Dataset root path - UPDATE THIS PATH")
    # 2023/09/18
    parser.add_argument('-chosen_clients', '--chosen_clients', action='store_true', default=False, help="True: perform training on certain client")
    # 2023/09/24
    parser.add_argument('--eval_mode', type=int, default=1, help="no client test(1) / *client test by utt(2) / client dev by utt(3)")
    # 2023/12/07
    parser.add_argument('-WeightedAvg', '--WeightedAvg', action='store_true', default=False, help="True: perform weighted sum when aggregating models")
    parser.add_argument('-CBFL', '--CBFL', action='store_true', default=False, help="True: perform CBFL (train with all client data)")
    # 2024/02/09
    parser.add_argument('--FL_type', type=int, default=1, help="FL(1) / FedProx(2) / FML(3)")
    parser.add_argument('-mu', '--mu', type=float, default=0.5, help="mu for FedProx")
    
    parser.add_argument('-alpha', '--alpha', type=float, default=0.5, help="alpha for FML for local")
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help="beta for FML for mutual")
    parser.add_argument('-FML_model', '--FML_model', type=int, help="local(0) or mutual(1). NOT TO BE ASSIGNED")

    # 2025/02/04 - Fuzzy clustering parameters
    parser.add_argument('--use_soft_clustering', action='store_true',help='Whether to use Fuzzy C-means instead of K-means for clustering')
    parser.add_argument('--fuzzy_m', type=float, default=2.0,help='Fuzzifier parameter m in Fuzzy C-means (usually between 1.5 to 3.0)')
    parser.add_argument('--membership_threshold', type=float, default=0.2,help='Minimum membership threshold for a sample to participate in cluster training')

    # 2025/02/17 - Membership weighted averaging
    parser.add_argument('--use_membership_weighted_avg', action='store_true', help='Whether to use membership values when averaging model weights')
    parser.add_argument('--membership_weight_factor', type=float, default=1.0, help='Factor to control the influence of membership values in weight averaging (0-1)')

    # MixSpeech data augmentation parameters
    parser.add_argument('--use_mixspeech', action='store_true', default=False, help='Whether to use MixSpeech data augmentation')
    parser.add_argument('--mixspeech_alpha', type=float, default=0.5, help='Beta distribution parameter for MixSpeech')
    parser.add_argument('--mixspeech_max_lambda', type=float, default=0.5, help='Maximum mixing weight for MixSpeech')
    parser.add_argument('--mixspeech_prob', type=float, default=0.15, help='Probability of applying MixSpeech')
    
    # Additional configurable parameters (previously hardcoded)
    parser.add_argument('--time_steps_median_supervised', type=int, default=130, help='Time steps median for supervised training')
    parser.add_argument('--time_steps_median_combined', type=int, default=149, help='Time steps median for combined dataset training')
 
    args = parser.parse_args()
    return args