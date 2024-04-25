""" 
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""

"""
import required libraries
"""
import numpy as np
import pandas as pd
from Bio import SeqIO
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import torch
from transformers import T5EncoderModel, T5Tokenizer, Trainer, TrainingArguments, EvalPrediction
import re
import gc
import esm
import ankh
from functools import partial
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn import metrics
from scipy import stats

"""
define file paths and other parameters
"""
input_fasta_file = "input/sequence.fasta" # load test sequence
output_csv_file = "output/results.csv" 
prot_t5_model_path = 'models/protT5_model_ann.h5'
esm2_model_path = 'models/esm3B_model_ann.h5'
ankh_model_path = 'models/ankh_model_ann.h5'

cutoff_threshold_ankh = 0.496
cutoff_threshold_esm = 0.797
cutoff_threshold_prott5 = 0.669


# create results dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'ankh_prob(Th = 0.496)','prot_t5_prob(Th = 0.669)','esm2_prob(Th = 0.797)', 'final_prediction'])


"""
Load tokenizer and pretrained model ProtT5, Ankh, ESM
"""
# install SentencePiece transformers if not installed already
#!pip install -q SentencePiece transformers

tokenizer_prot_t5 = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
pretrained_model_prot_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

# pretrained_model = pretrained_model.half()
gc.collect()

# define devices
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
pretrained_model_prot_t5 = pretrained_model_prot_t5.to(device)
pretrained_model_prot_t5 = pretrained_model_prot_t5.eval()


# To load ankh model:
ankh_pretrained_model, tokenizer_ankh = ankh.load_large_model()
ankh_pretrained_model.eval()


# Load ESM-2 model
pretrained_model_esm, alphabet_esm = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter_esm = alphabet_esm.get_batch_converter()
pretrained_model_esm.eval()  # disables dropout for deterministic results



def get_1021_string(sequence, site=0):
    
    """ 
        We are taking one sequence at a time because of the memory issue, this can be improved a lot
    """
    
    # truncate sequence to peptide of 1024 if it is greater
    if len(sequence) > 1021:
        if site < 511:
            # take first 1023 window
            sequence_truncated = sequence[:1021]
            new_site = site
            
        elif site > len(sequence)-511:
            sequence_truncated = sequence[-1021:]
            new_site = 1021 - (len(sequence) - site) 
        else:
            # Use new position just to extract the feature, store original 
            sequence_truncated = sequence[site - 510 : site + 510 + 1]
            new_site = 510
    else: 
        # change nothing
        new_site = site
        sequence_truncated = sequence
        
    return new_site, sequence_truncated

def get_ankh_features(sequence):

    def embed_dataset(model, sequences, shift_left = 0, shift_right = -1):
        inputs_embedding = []
        with torch.no_grad():
            ids = tokenizer_ankh.batch_encode_plus([[sequence]], 
                                              add_special_tokens=True, 
                                              padding=True,
                                              return_tensors="pt",
                                              is_split_into_words=True
                                             )
            embedding = model(input_ids=ids['input_ids'].to(device))[0]
            embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
        return embedding
    
    def preprocess_dataset(sequences, max_length=None):
        '''
            Args:
                sequences: list, the list which contains the protein primary sequences.
                labels: list, the list which contains the dataset labels.
                max_length, Integer, the maximum sequence length, 
                if there is a sequence that is larger than the specified sequence length will be post-truncated. 
        '''
        if max_length is None:
            max_length = len(max(sequences, key=lambda x: len(x)))
        splitted_sequences = [list(seq[:max_length]) for seq in sequences]
        return splitted_sequences

    training_embeddings = embed_dataset(ankh_pretrained_model, preprocess_dataset([sequence]))  
    
    return training_embeddings
    
    
def get_protT5_features(sequence): 
    
    """
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Input:
        sequence (str): str of length l
    Returns:
        tensor: l*1024
    """
    
    # replace rare amino acids with X
    sequence = re.sub(r"[UZOB]", "X", sequence)
    
    # add space in between amino acids
    sequence = [ ' '.join(sequence)]
    
    # set configurations and extract features
    ids = tokenizer_prot_t5.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding = pretrained_model_prot_t5(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()
    
    # find length
    seq_len = (attention_mask[0] == 1).sum()
    
    # select features
    seq_emd = embedding[0][:seq_len-1]
    
    return seq_emd


def get_esm2_3B_features(sequence):
  
    # prepare input df in the format that model accepts
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    # ]
    
    # prepare dataframe of truncated string
    data = [
        ("pid", sequence),
    ]
    
    batch_labels, batch_strs, batch_tokens = batch_converter_esm(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = pretrained_model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
  
    # return only residue level embeddings so that we can treat them exactly as prott5 features that we are already using
    return token_representations[:,1:-1,:][0]



# main function--------------------------------------------------------------------------------
# load models
prot_t5_model_ann = load_model(prot_t5_model_path,compile=False)
esm2_model_ann = load_model(esm2_model_path,compile=False)
ankh_model_ann = load_model(ankh_model_path,compile=False)

for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = str(seq_record.seq)
    
    # if sequence is longer than 1021, truncate
    if len(sequence) > 1021:
        sequence = get_1021_string(sequence)
    
    positive_predicted = []
    negative_predicted = []
    
    # extract protT5 for full sequence and store in temporary dataframe 
    pt5_all = get_protT5_features(sequence)
    ankh_all = get_ankh_features(sequence)
    esm_all = get_esm2_3B_features(sequence)
    
    # generate embedding features and window for each amino acid in sequence
    for index, amino_acid in enumerate(sequence):
        
        # check if AA is 'S' or 'T'
        if amino_acid in ['S', 'T']:
            
            # we consider site one more than index, as index starts from 0
            site = index + 1

            # get ProtT5, ESM, ANN features extracted above
            X_test_pt5 = pt5_all[index]
            X_test_esm = esm_all[index]
            X_test_ankh = ankh_all[index]
            
            # prediction results           
            y_pred_ankh = ankh_model_ann.predict(np.array(X_test_ankh.reshape(1,1536)), verbose = 0)[0][0]
            y_pred_esm = esm2_model_ann.predict(np.array(X_test_esm.reshape(1,2560)), verbose = 0)[0][0]
            y_pred_prot_t5 = prot_t5_model_ann.predict(np.array(X_test_pt5.reshape(1,1024)), verbose = 0)[0][0]
            
            # combined result
            voting_result = int(y_pred_ankh > cutoff_threshold_ankh) + int(y_pred_esm > cutoff_threshold_esm) + int(y_pred_prot_t5 > cutoff_threshold_prott5)
            
            # append results to results_df
            results_df.loc[len(results_df)] = [prot_id, site, amino_acid, y_pred_ankh, y_pred_prot_t5, y_pred_esm, int(voting_result > 1) ]

# Export results 
print('Saving results ...')
results_df.to_csv(output_csv_file, index = False)
print('Results saved to ' + output_csv_file)
