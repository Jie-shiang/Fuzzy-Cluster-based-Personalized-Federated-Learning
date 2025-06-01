from update_CPFL import ASRLocalUpdate_CPFL
import torch, os

def client_train(args, model_in_path_root, model_out_path, train_dataset_supervised, 
                 test_dataset, idx, epoch, cluster_id, global_weights=None):
    torch.set_num_threads(1)
    
    # Import logging function
    from federated_main import exp_log
    
    if epoch == 0:  # First round training - use global model
        if "_FLASR" in model_in_path_root:
            model_in_path = model_in_path_root + "_global/final/"
        else:
            model_in_path = model_in_path_root + "_FLASR_global/final/"
        
        exp_log(f"First round training using global model path: {model_in_path}")
        
        if not os.path.exists(model_in_path):
            exp_log(f"Warning: Global model path does not exist: {model_in_path}")
            
    else:  # Non-first round - use previous client model
        if args.training_type == 1:  # supervised
            clean_model_out_path = model_out_path.replace("_FLASR", "")
            
            model_in_path = clean_model_out_path + "_client" + str(idx) + "_round" + str(epoch-1)
            if cluster_id != None:
                model_in_path += "_cluster" + str(cluster_id) 
            
            model_in_path += "_TrainingAddress/final/"
            
            exp_log(f"Round {epoch} training using client model path: {model_in_path}")
            
            if not os.path.exists(model_in_path):
                exp_log(f"Warning: Client model path does not exist: {model_in_path}")
    
    local_model = ASRLocalUpdate_CPFL(args=args, dataset_supervised=train_dataset_supervised, global_test_dataset=test_dataset, client_id=idx, 
                                      cluster_id=cluster_id, model_in_path=model_in_path, model_out_path=model_out_path)
                                                                                      
    w, num_training_samples = local_model.update_weights(global_weights=global_weights, global_round=epoch) 
                                                                                    
    torch.cuda.empty_cache()
    return w, num_training_samples

def centralized_training(args, model_in_path, model_out_path, train_dataset, test_dataset, epoch, client_id="public"):                    
    # Training function for global model, train from model in model_in_path
    # Final result in model_out_path + "_global/final"
    local_model = ASRLocalUpdate_CPFL(args=args, dataset_supervised=train_dataset, global_test_dataset=test_dataset, client_id=client_id, 
                        cluster_id=None, model_in_path=model_in_path+"final/", model_out_path=model_out_path)   
    # Initialize public dataset
    local_model.update_weights(global_weights=None, global_round=epoch)               # From model_in_path to train


def client_getEmb(args, model_in_path, train_dataset_supervised, test_dataset, idx, cluster_id, TEST):
   # Function to get embeddings for each client, from model in model_in_path +"/final/"
   torch.set_num_threads(1)
   local_model = ASRLocalUpdate_CPFL(args=args, 
                                   dataset_supervised=train_dataset_supervised,
                                   global_test_dataset=test_dataset,
                                   client_id=idx,
                                   cluster_id=cluster_id,
                                   model_in_path=model_in_path,
                                   model_out_path=None)
   
   # Initialize dataset of current client & current cluster
   if TEST:
       df, hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features = local_model.extract_embs(TEST)
       return df
   else:
       hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features = local_model.extract_embs(TEST)
       return hidden_states_mean, loss, entropy, vocab_ratio_rank, encoder_attention_1D, speech_features