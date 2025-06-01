def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Current Stage   : {args.FL_STAGE}\n')

    print('    Federated parameters:')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Eval step is set to  : {args.eval_steps}')
    print(f'    Current training type: {args.training_type}')
    print(f'    Current number of clusters: {args.num_lms}')

    return

def add_cluster_id(example, cluster_info):
    """
    Add cluster information to sample
    Args:
        example: Data sample
        cluster_info: Cluster ID or membership information
    """
    if isinstance(cluster_info, (int, np.integer)):
        # Traditional K-means case
        example["cluster_id"] = cluster_info
    elif isinstance(cluster_info, (list, np.ndarray)):
        # FCM case
        example["cluster_memberships"] = cluster_info
        example["cluster_id"] = int(np.argmax(cluster_info))
    return example

def gen_mapping_fn(args, processor, model_lst):
    def map_to_result(batch):                                               # 1 sample per batch
        with torch.no_grad():
            if args.num_lms > 1:                                            # for multi-cluster
                model_id = batch["cluster_id"]                              # get cluster_id for this sample
                model = model_lst[model_id]                                 # use corresponding model
            else:
                model = model_lst[0]                                        # use the 1st model for uni-cluster
            # decode using corresponding model
            input_values = torch.tensor(batch["input_values"]).unsqueeze(0).to("cuda")
            model = model.to("cuda")
            logits = model(input_values).logits
            # save result
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids)[0]
            batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
        return batch
    return map_to_result

# WER computation functions (keeping original jiwer-based implementation)
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import Levenshtein
from jiwer import transforms as tr
from jiwer.transformations import wer_default, wer_standardize

from itertools import chain

def _is_list_of_list_of_strings(x: Any, require_non_empty_lists: bool):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if require_non_empty_lists and len(e) == 0:
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True
    
def _preprocess(
    truth: List[str],
    hypothesis: List[str],
    truth_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
) -> Tuple[List[str], List[str]]:
    """
    Pre-process the truth and hypothesis into a form such that the Levenshtein
    library can compute the edit operations.can handle.
    """
    # Apply transforms. The transforms should collapses input to a list of list of words
    transformed_truth = truth_transform(truth)
    transformed_hypothesis = hypothesis_transform(hypothesis)

    # raise an error if the ground truth is empty or the output
    # is not a list of list of strings
    if len(transformed_truth) != len(transformed_hypothesis):
        raise ValueError(
            "number of ground truth inputs ({}) and hypothesis inputs ({}) must match.".format(
                len(transformed_truth), len(transformed_hypothesis)
            )
        )
    if not _is_list_of_list_of_strings(transformed_truth, require_non_empty_lists=True):
        raise ValueError(
            "truth should be a list of list of strings after transform which are non-empty"
        )
    if not _is_list_of_list_of_strings(
        transformed_hypothesis, require_non_empty_lists=False
    ):
        raise ValueError(
            "hypothesis should be a list of list of strings after transform"
        )

    # tokenize each word into an integer
    vocabulary = set(chain(*transformed_truth, *transformed_hypothesis))

    if "" in vocabulary:
        raise ValueError(
            "Empty strings cannot be a word. "
            "Please ensure that the given transform removes empty strings."
        )

    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    truth_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in transformed_truth
    ]
    hypothesis_chars = [
        "".join([chr(word2char[w]) for w in sentence])
        for sentence in transformed_hypothesis
    ]

    return truth_chars, hypothesis_chars

def _get_operation_counts(
    source_string: str, destination_string: str
) -> Tuple[int, int, int, int]:
    """
    Check how many edit operations (delete, insert, replace) are required to
    transform the source string into the destination string.
    """
    editops = Levenshtein.editops(source_string, destination_string)
            
    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)
    
    return hits, substitutions, deletions, insertions

def compute_measures(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate error measures between a set of ground-truth sentences and a set of
    hypothesis sentences.
    """
    # validate input type
    if isinstance(truth, str):
        truth = [truth]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in truth):
        raise ValueError("one or more groundtruths are empty strings")

    # Preprocess truth and hypothesis
    trans = truth
    pred = hypothesis

    truth, hypothesis = _preprocess(
        truth, hypothesis, truth_transform, hypothesis_transform
    )

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    H, S, D, I = 0, 0, 0, 0

    # also keep track of the total number of ground truth words and hypothesis words
    gt_tokens, hp_tokens = 0, 0
    
    i = 0
    for groundtruth_sentence, hypothesis_sentence in zip(truth, hypothesis):
        # Get the operation counts (#hits, #substitutions, #deletions, #insertions)       
        hits, substitutions, deletions, insertions = _get_operation_counts(
            groundtruth_sentence, hypothesis_sentence
        )

        H += hits
        S += substitutions
        D += deletions
        I += insertions
        gt_tokens += len(groundtruth_sentence)
        hp_tokens += len(hypothesis_sentence)
        i = i + 1

    # Compute Word Error Rate
    wer = float(S + D + I) / float(H + S + D)

    # Compute Match Error Rate
    mer = float(S + D + I) / float(H + S + D + I)

    # Compute Word Information Preserved
    wip = (float(H) / gt_tokens) * (float(H) / hp_tokens) if hp_tokens >= 1 else 0

    # Compute Word Information Lost
    wil = 1 - wip        

    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "hits": H,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
    }

def record_WER(args, result, cluster_num, test_data="global"):
    wer_result = compute_measures(truth=result["text"], hypothesis=result["pred_str"])
    # Filter out different classes
    class_0_result = result.filter(lambda example: example.get("class_labels", example.get("dementia_labels", 0))==0 and example['text'] != '')
    class_1_result = result.filter(lambda example: example.get("class_labels", example.get("dementia_labels", 0))==1 and example['text'] != '')

    if len(class_0_result["text"]) != 0:                                         # if sample exists, compute wer
        wer_class_0 = compute_measures(truth=class_0_result["text"], hypothesis=class_0_result["pred_str"])
    else:
        wer_class_0 = {"wer": "No sample"}                                       # or record "No sample"

    if len(class_1_result["text"]) != 0:                                         # if sample exists, compute wer
        wer_class_1 = compute_measures(truth=class_1_result["text"], hypothesis=class_1_result["pred_str"])
    else:
        wer_class_1 = {"wer": "No sample"}                                       # or record "No sample"

    if cluster_num != None:                                                 # record cluster_id if given
        model_name = args.model_in_path.split("/")[-1] + "_cluster" + str(cluster_num)
    else:
        model_name = args.model_in_path.split("/")[-1]

    model_name = model_name + "_" + test_data

    data = {
    'model': model_name,
    'WER': [wer_result["wer"]],
    'Class_1_WER': [wer_class_1["wer"]],  # MODIFIED: Generic class names
    'Class_0_wer': [wer_class_0["wer"]],  # MODIFIED: Generic class names
    'HITS': [wer_result["hits"]],
    'substitutions': [wer_result["substitutions"]],
    'deletions': [wer_result["deletions"]],
    'insertions': [wer_result["insertions"]]
    }
    df = pd.DataFrame(data)

    # check if file exists
    file_exists = os.path.isfile('./results/Overall_WER.csv')

    # if file exists, no header
    if file_exists:
        df.to_csv('./results/Overall_WER.csv', mode='a', header=False, index=False)
    else:
        # create new file
        df.to_csv('./results/Overall_WER.csv', index=False)

def get_overall_wer(args, dataset, test_data="global"):
    torch.set_num_threads(1)
    # load ASR model
    mask_time_prob = 0                                                      # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)

    model_lst = []
    if args.num_lms > 1:                                                    # multi-cluster
        for cluster_id in range(args.num_lms):                              # load model 1 by 1
            txt = args.model_in_path.split("#")
            model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
            model_lst.append(model)
    else:                                                                   # load from args.model_in_path
        model = load_model(args, args.model_in_path, config)
        model_lst.append(model)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)

    result = dataset.map(gen_mapping_fn(args, processor, model_lst))
    record_WER(args, result, None, test_data=test_data)

def load_model(args, model_in_path, config):                                # model_in_path w.o. "/final/"
    file_to_check = model_in_path + "/decoder_weights.pth"
    if os.path.isfile(file_to_check):
        # file exists
        # MODIFIED: Use configurable path instead of hardcoded path
        global_model_path = args.model_out_path.replace(args.model_out_path.split("/")[-1], "") + "data2vec-audio-large-960h_FLASR_global/final"
        model = Data2VecAudioForCTC_CPFL.from_pretrained(global_model_path, config=config, args=args)
        # load decoder's weight
        decoder_state_dict = torch.load(model_in_path + "/decoder_weights.pth")
        model.lm_head.load_state_dict(decoder_state_dict)
    else:
        model = Data2VecAudioForCTC_CPFL.from_pretrained(model_in_path+"/final/", config=config, args=args)
    
    return model

############################################################################################
# Splits-related: client, train / test
############################################################################################
# for each speaker type, take 80% of data as training and 20% as testing
def split_train_test_spk(source_dataset, client_spk, identity, DEV):
    # identity: speaker type identifier (e.g., "INV" or "PAR")
    subsetA = source_dataset.filter(lambda example: example["path"].startswith(client_spk+"_"+identity))
                                                    # filter out a single spk
    subsetA = subsetA.sort("path")                  
    LEN_subsetA = len(subsetA)                      # num of sample for this spk
    if DEV:
        num_sample_train = max(1, int(LEN_subsetA*0.7))       # min 1, use 70% of samples as training
        num_sample_trainDev = max(1, int(LEN_subsetA*0.8))    # min 1, use 80% as (training + dev)
        
        if num_sample_train == 0:                             # if 0 sample
            train_dataset = Dataset.from_dict({})             # return empty dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample_train))
                                                              # select 70% as training
        if num_sample_train == num_sample_trainDev:           # if 0 sample
            test_dataset = Dataset.from_dict({})              # return empty dataset
        else:        
            test_dataset = subsetA.select(range(num_sample_train, num_sample_trainDev))
                                                              # select 10% as dev
    else:
        num_sample = max(1, int(LEN_subsetA*0.8))             # min 1, use 80% as training
        
        if num_sample == 0:                                   # if 0 sample
            train_dataset = Dataset.from_dict({})             # return empty dataset
        else:
            train_dataset = subsetA.select(range(0, num_sample))
                                                              # select 80% as training
        if num_sample == LEN_subsetA:                         # if 0 sample
            test_dataset = Dataset.from_dict({})              # return empty dataset
        else:        
            test_dataset = subsetA.select(range(num_sample, LEN_subsetA))
                                                              # select 20% as testing
    return train_dataset, test_dataset

from datasets import concatenate_datasets
def concatenate_ds(datasetA, datasetB):
    if len(datasetA) != 0 and len(datasetB) != 0:             # if both non-empty, combine them
        concatenated_dataset = concatenate_datasets([datasetA, datasetB])
        return concatenated_dataset
    
    # at least one of them is empty
    if len(datasetA) != 0:                                    # A not empty, return it
        return datasetA    
    return datasetB                                           # return B

# return train / test set of this client
def split_train_test_client(client_spks, source_dataset, DEV=False):    
                                                              # default: no dev
    # for 1st spk_id, get training(80%) and testing(20%) data for different speaker types
    client_spk = client_spks[0]
    train_dataset_type1, test_dataset_type1 = split_train_test_spk(source_dataset, client_spk, "INV", DEV)  # MODIFY AS NEEDED
    train_dataset_type2, test_dataset_type2 = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)  # MODIFY AS NEEDED

    # combine different types
    train_dataset_client = concatenate_ds(train_dataset_type1, train_dataset_type2)
    test_dataset_client = concatenate_ds(test_dataset_type1, test_dataset_type2)

    for i in range(len(client_spks)-1):                       # for each spk_id
        # get training(80%) and testing(20%) data for different speaker types
        client_spk = client_spks[i+1]
        train_dataset_type1, test_dataset_type1 = split_train_test_spk(source_dataset, client_spk, "INV", DEV)
        train_dataset_type2, test_dataset_type2 = split_train_test_spk(source_dataset, client_spk, "PAR", DEV)

        # combine types
        train_dataset_spk = concatenate_ds(train_dataset_type1, train_dataset_type2)
        test_dataset_spk = concatenate_ds(test_dataset_type1, test_dataset_type2)

        # combine to client data
        train_dataset_client = concatenate_ds(train_dataset_client, train_dataset_spk)
        test_dataset_client = concatenate_ds(test_dataset_client, test_dataset_spk)    

    return train_dataset_client, test_dataset_client

# REMOVED: Hardcoded speaker-to-client mapping
# TODO: Update this function based on your dataset's speaker IDs
def client2spk(client_id):
    """
    Map client ID to speaker ID(s)
    TODO: Update this mapping based on your dataset
    """
    # Example mapping - replace with your actual speaker IDs
    client2spk_dict = {
        '1': 'SPEAKER_001', '2': 'SPEAKER_002', '3': 'SPEAKER_003',
        # Add more mappings as needed
    }
    return [client2spk_dict.get(str(client_id+1), f'SPEAKER_{client_id+1:03d}')]
    
# Mode 1: no client test
# Mode 2: client test by utt
# Mode 3: client dev by utt
def train_split_supervised(args, dataset, client_id, cluster_id):
    # generate sub- training set for given user-ID
    if args.num_users > 30:                                                                 # for "spk as client" setting
        client_spks = client2spk(client_id)
        print("Current spk: ", client_spks)
    elif client_id == "public":
        # REMOVED: Hardcoded speaker list
        # TODO: Update this list based on your dataset
        client_spks = [
            # Add your speaker IDs here
            'SPEAKER_001', 'SPEAKER_002', 'SPEAKER_003'  # EXAMPLE - replace with actual IDs
        ]
    elif client_id == "public2":
        # REMOVED: Hardcoded speaker list
        # TODO: Update this list based on your dataset
        client_spks = [
            # Add your speaker IDs here
            'SPEAKER_004', 'SPEAKER_005', 'SPEAKER_006'  # EXAMPLE - replace with actual IDs
        ]
        print("Train with all client data")
    elif client_id == 0:
        client_spks = ['SPEAKER_001', 'SPEAKER_002']  # EXAMPLE - replace with actual IDs
    elif client_id == 1: 
        client_spks = ['SPEAKER_003', 'SPEAKER_004']  # EXAMPLE - replace with actual IDs
    elif client_id == 2:  
        client_spks = ['SPEAKER_005', 'SPEAKER_006']  # EXAMPLE - replace with actual IDs
    elif client_id == 3:   
        client_spks = ['SPEAKER_007', 'SPEAKER_008']  # EXAMPLE - replace with actual IDs
    elif client_id == 4: 
        client_spks = ['SPEAKER_009', 'SPEAKER_010']  # EXAMPLE - replace with actual IDs
    else:
        print("Train with whole dataset!!")
        return dataset

    print(f"Generating client training set for client {str(client_id)}...")
    
    # 2. Filter data based on speaker
    client_dataset = dataset.filter(
        lambda example: example["path"].startswith(tuple(client_spks))
    )
    
    # 3. Further filter data if cluster_id is specified
    if cluster_id is not None:
        print(f"Processing data for cluster {cluster_id}...")
        if args.use_soft_clustering:
            # Use membership filtering
            client_dataset = client_dataset.filter(
                lambda example: (
                    'cluster_memberships' in example and 
                    example['cluster_memberships'][cluster_id] >= args.membership_threshold
                )
            )
            print(f"Selected {len(client_dataset)} samples with membership >= {args.membership_threshold}")
        else:
            # Traditional hard clustering
            client_dataset = client_dataset.filter(
                lambda example: example['cluster_id'] == cluster_id
            )
            print(f"Selected {len(client_dataset)} samples for cluster {cluster_id}")
    
    # 4. Return based on eval_mode
    if args.eval_mode == 1:
        return client_dataset, None
    elif args.eval_mode == 2:
        return split_train_test_client(client_spks, client_dataset)
    elif args.eval_mode == 3:
        return split_train_test_client(client_spks, client_dataset, DEV=True)
    
    return client_dataset, None

def compute_utterance_measures(truth: str, hypothesis: str):
    """
    Compute WER measures for a single utterance using the same method as in compute_measures
    """
    # Preprocess strings using the same transforms as in compute_measures
    truth_chars, hypothesis_chars = _preprocess(
        [truth], [hypothesis], 
        truth_transform=wer_default,
        hypothesis_transform=wer_default
    )
    
    # Get operation counts for this utterance
    hits, substitutions, deletions, insertions = _get_operation_counts(
        truth_chars[0], hypothesis_chars[0]
    )
    
    # Compute metrics
    N = len(truth_chars[0])  # total length of reference
    wer = float(substitutions + deletions + insertions) / float(hits + substitutions + deletions)
    mer = float(substitutions + deletions + insertions) / float(hits + substitutions + deletions + insertions)
    
    # Word Information Preserved/Lost
    hp_tokens = len(hypothesis_chars[0])
    wip = (float(hits) / N) * (float(hits) / hp_tokens) if hp_tokens >= 1 else 0
    wil = 1 - wip

    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions
    }

def evaluateASR(args, global_round, global_test_dataset, train_dataset_supervised=None):
    """
    Evaluate ASR performance and record both overall and (if final round) detailed results
    """
    if args.chosen_clients == True:  
        idxs_users = [0, 4]
    else:
        idxs_users = range(args.num_users)  

    # First part: evaluate individual client models (keep original evaluation logic)
    for i in idxs_users:  
        save_path = args.model_out_path + "_client" + str(i) + "_round" + str(global_round)
        if args.num_lms > 1:  
            save_path += "#"
        if args.training_type == 1:  
            save_path += "_Training" + "Address"
        if args.FL_type == 3:  
            save_path += "_localModel"  
        args.model_in_path = save_path
        
        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        mask_time_prob = 0
        config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
        
        model_lst = []
        if args.num_lms > 1:
            for cluster_id in range(args.num_lms):
                txt = args.model_in_path.split("#")
                model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
                model_lst.append(model)
        else:
            model = load_model(args, args.model_in_path, config)
            model_lst.append(model)
            
        result = global_test_dataset.map(gen_mapping_fn(args, processor, model_lst))
        record_WER(args, result, None)
        
        if args.eval_mode == 2 or args.eval_mode == 3:  
            origin_eval_mode = args.eval_mode  
            args.eval_mode = 2  
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)
            if len(test_dataset_client) == 0:  
                test_result = global_test_dataset.map(gen_mapping_fn(args, processor, model_lst))
                record_WER(args, test_result, None)
            else:
                test_result = test_dataset_client.map(gen_mapping_fn(args, processor, model_lst))
                record_WER(args, test_result, None, test_data="test")
            args.eval_mode = origin_eval_mode
        
        if args.eval_mode == 3:  
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)
            if len(test_dataset_client) == 0:  
                test_result = global_test_dataset.map(gen_mapping_fn(args, processor, model_lst))
                record_WER(args, test_result, None)
            else:
                test_result = test_dataset_client.map(gen_mapping_fn(args, processor, model_lst))
                record_WER(args, test_result, None, test_data="dev")

    # Second part: evaluate aggregated model
    if args.num_lms > 1:  
        args.model_in_path = args.model_out_path+"#_CPFLASR_global_round" + str(global_round)
    else:  
        args.model_in_path = args.model_out_path+"_cluster0_CPFLASR_global_round" + str(global_round)
    
    # Load aggregated model
    processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
    mask_time_prob = 0
    config = Data2VecAudioConfig.from_pretrained(args.pretrain_name, mask_time_prob=mask_time_prob)
    
    model_lst = []
    if args.num_lms > 1:
        for cluster_id in range(args.num_lms):
            txt = args.model_in_path.split("#")
            model = load_model(args, txt[0] + "_cluster" + str(cluster_id) + txt[1], config)
            model_lst.append(model)
    else:
        model = load_model(args, args.model_in_path, config)
        model_lst.append(model)

    # Test with aggregated model
    result = global_test_dataset.map(gen_mapping_fn(args, processor, model_lst))
    record_WER(args, result, None)

    if args.eval_mode == 2 or args.eval_mode == 3:
        # In final round, record detailed results
        if global_round == args.epochs - 1:
            data = {
                'path': [],
                'text': [],
                'class_labels': [],  # MODIFIED: Generic labels
                'cluster_id': [],
                'pred_str': [],
                'WER': [],
                'HITS': [],
                'substitutions': [],
                'deletions': [],
                'insertions': [],
                'client_id': []
            }
            
            # If using soft clustering, add related fields
            if args.use_soft_clustering:
                data.update({
                    'cluster_memberships': [],
                    'valid_clusters': [],
                    'normalized_memberships': []
                })
        
        # Test each client's test set using aggregated model
        for i in idxs_users:
            _, test_dataset_client = train_split_supervised(args, train_dataset_supervised, client_id=i, cluster_id=None)
            
            if len(test_dataset_client) > 0:
                # Use aggregated model for prediction
                test_result = test_dataset_client.map(gen_mapping_fn(args, processor, model_lst))
                record_WER(args, test_result, None, test_data="test")
                
                # If final round, collect detailed results
                if global_round == args.epochs - 1:
                    for item in test_result:
                        # Add basic information
                        data['path'].append(item['path'])
                        data['text'].append(item['text'])
                        data['class_labels'].append(item.get('class_labels', item.get('dementia_labels', 0)))
                        data['cluster_id'].append(item.get('cluster_id', 'None'))
                        data['pred_str'].append(item['pred_str'])
                        data['client_id'].append(i)
                        
                        # Calculate WER metrics
                        single_result = compute_measures(truth=[item['text']], hypothesis=[item['pred_str']])
                        data['WER'].append(single_result['wer'])
                        data['HITS'].append(single_result['hits'])
                        data['substitutions'].append(single_result['substitutions'])
                        data['deletions'].append(single_result['deletions'])
                        data['insertions'].append(single_result['insertions'])
                        
                        # If using soft clustering, add related information
                        if args.use_soft_clustering:
                            data['cluster_memberships'].append(item.get('cluster_memberships', []))
                            data['valid_clusters'].append(item.get('valid_clusters', []))
                            data['normalized_memberships'].append(item.get('normalized_memberships', []))
        
        # If final round, save all detailed results
        if global_round == args.epochs - 1:
            df = pd.DataFrame(data)
            df.to_csv('./results/Detailed_WER_round9.csv', index=False)
            
            # If using soft clustering, additionally save clustering analysis
            if args.use_soft_clustering:
                cluster_stats = {
                    'cluster_id': [],
                    'total_samples': [],
                    'avg_membership': [],
                    'valid_samples': []
                }
                
                for c in range(args.num_lms):
                    memberships = [m[c] for m in data['cluster_memberships'] if len(m) > c]
                    valid_samples = sum(1 for m in memberships if m >= args.membership_threshold)
                    
                    cluster_stats['cluster_id'].append(c)
                    cluster_stats['total_samples'].append(len(memberships))
                    cluster_stats['avg_membership'].append(np.mean(memberships))
                    cluster_stats['valid_samples'].append(valid_samples)
                
                pd.DataFrame(cluster_stats).to_csv('./results/Cluster_Stats_round9.csv', index=False)
                #!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import time
import torch

from transformers import Wav2Vec2Processor
from datasets import Dataset
import librosa
import numpy as np
import pandas as pd
import os
from datasets import load_from_disk
import scipy
import argparse
from datasets import * 
from transformers import Data2VecAudioConfig
from models import Data2VecAudioForCTC_CPFL
from jiwer import wer
import warnings
import re

# REMOVED: Environment variables for hardcoded paths
# TODO: Update these paths based on your project structure
# CPFL_codeRoot = os.environ.get('CPFL_codeRoot')
# CPFL_dataRoot = os.environ.get('CPFL_dataRoot')
CPFL_codeRoot = './'  # UPDATE THIS PATH
CPFL_dataRoot = './data'  # UPDATE THIS PATH

# some parameters
parser = argparse.ArgumentParser()
parser.add_argument('-opt', '--optimizer', type=str, default="adamw_hf", help="The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor")
parser.add_argument('-MGN', '--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm (for gradient clipping)")
parser.add_argument('-model_type', '--model_type', type=str, default="data2vec", help="Type of the model")
parser.add_argument('-sr', '--sampl_rate', type=float, default=16000, help="librosa read sampling rate")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning rate")
# MODIFIED: Use relative path instead of absolute path
parser.add_argument('-RD', '--root_dir', default='./data', help="Root directory - UPDATE THIS PATH")
parser.add_argument('--AudioLoadFunc', default='librosa', help="scipy function might perform faster")
args = parser.parse_args(args=[])

def prepare_dataset(batch, processor, with_transcript=True):
    if "input_values" not in batch.keys():                                  # get input_values only for the 1st time
        audio = batch["array"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        
    if with_transcript:                                                     # if given transcript
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids            # generate labels
        
    return batch

def ID2Label(ID, spk2label):
    name = ID.split("_")                                                    # from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is healthy (not pathological)
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return class label for this file

def csv2dataset(audio_path = None, csv_path = None, dataset_path = "./dataset/", with_transcript=True):
    """
    MODIFIED: Removed hardcoded paths, now requires explicit paths
    TODO: Update the paths when calling this function
    """
    if audio_path is None:
        audio_path = f'{args.root_dir}/clips/'  # UPDATE THIS PATH
    if csv_path is None:
        csv_path = f'{args.root_dir}/train.csv'  # UPDATE THIS PATH
        
    stored = dataset_path + csv_path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(csv_path)                                            # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset

    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    if with_transcript:
        my_dict["text"] = []                                                # ground truth transcript if given
    my_dict["class_labels"] = []  # MODIFIED: Use generic labels

    # REMOVED: Hardcoded dictionary path
    # TODO: Update this path to your label dictionary
    path2_dict = "./data/label_dict.npy"  # UPDATE THIS PATH

    if with_transcript:                                                     
        if os.path.exists(path2_dict):
            spk2label = np.load(path2_dict, allow_pickle=True).tolist()
        else:
            print(f"Warning: Label dictionary not found at {path2_dict}")
            spk2label = {}

    i = 1
    for file_path in dataset['path']:                                       # for all files
        if 'sentence' in dataset.features:                                  # if col "sentence" exists
            if dataset['sentence'][i-1] == None:                            # but no info
                i += 1                                                      # skip to next file
                continue                                                    # skip to next file
        if args.AudioLoadFunc == 'librosa':
            try:
                sig, s = librosa.load('{0}/{1}'.format(audio_path,file_path), sr=args.sampl_rate, dtype='float32')  
                                                                            # read audio w/ 16k sr
            except ValueError:                                            # skip files that can't be loaded                                                 
                print("Error file = ", audio_path,file_path)
        else:
            s, sig = scipy.io.wavfile.read('{0}/{1}'.format(audio_path,file_path))
            sig=librosa.util.normalize(sig)
        if len(sig) > 1600:                                                 # get rid of audio that's too short
            my_dict["path"].append(file_path)                               # add path
            my_dict["array"].append(sig)                                    # add audio wave
            if with_transcript:
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
            my_dict["class_labels"].append(ID2Label(ID=file_path, spk2label=spk2label))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There are ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset

def get_raw_dataset(args):                                                  # return whole training & testing set
    if args.dataset == 'pathological_speech':                              # MODIFIED: Generic dataset name
        if args.FL_STAGE == 4:
            dataset_path = args.dataset_path_root + "/clustered/"           # load clustered dataset
        else:
            dataset_path = args.dataset_path_root + "/"                     # load dataset w.o. cluster info
        
        processor = Wav2Vec2Processor.from_pretrained(args.pretrain_name)
        
        # load and map train data
        train_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/train.csv", dataset_path=dataset_path)
        train_dataset = train_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)

        # load and map test data
        test_data = csv2dataset(csv_path = f"{CPFL_dataRoot}/test.csv", dataset_path=dataset_path)
        test_dataset = test_data.map(lambda x: prepare_dataset(x, processor=processor), num_proc=10)

    return train_dataset, test_dataset

def reorder_col(datasetA, datasetB):                                        # order B as A, return re-ordered B
    # turn target Dataset to dataframe
    dfB = datasetB.to_pandas()

    # order B as A
    column_order = datasetA.column_names                                    # A's col order
    dfB = dfB[column_order]

    datasetB_reordered = Dataset.from_pandas(dfB)                           # turn back to type 'Dataset'
    return datasetB_reordered

def average_weights(w, num_training_samples_lst, membership_weights_lst=None, args=None):
    """
    Aggregate model weights from multiple clients
    Args:
        w: List of client model weights
        num_training_samples_lst: List of sample counts per client
        membership_weights_lst: List of average membership values per client
        args: Parameter settings
    Returns:
        w_avg: Aggregated weights
    """
    w_avg = copy.deepcopy(w[0])
    
    if not args.WeightedAvg:
        # Use simple averaging
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    
    # Calculate aggregation weights
    if args.use_membership_weighted_avg and membership_weights_lst is not None:
        # Combine sample counts and membership weights
        sample_weights = np.array(num_training_samples_lst)
        membership_weights = np.array(membership_weights_lst)
        
        # Use membership_weight_factor to control membership influence
        combined_weights = sample_weights * (
            (1.0 - args.membership_weight_factor) + 
            args.membership_weight_factor * membership_weights
        )
    else:
        # Only use sample counts
        combined_weights = np.array(num_training_samples_lst)
    
    # Normalize weights
    normalized_weights = combined_weights / combined_weights.sum()
    
    # Aggregate weights
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * 0  # Clear
        for i in range(len(w)):
            w_avg[key] += w[i][key] * normalized_weights[i]
    
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Current Stage   : {args.FL_STAGE}\n')

    print('    Federated parameters:')
    print(f'    Number of users    : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'