""" 
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from Bio import SeqIO
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     LeakyReLU, MaxPooling1D, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


def get_protT5_features(sequence):
    # THIS NEEDS TO BE REPLACED BY PROTT5 FEATURE EXTRACTION CODE
    dummy = np.array([np.random.uniform(size = (len(sequence),1024), low = 0, high = 1)])[0]
    print(dummy.shape)
    return dummy

# load test sequence
fasta_file = "input/sequence.fasta"

# create results dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'probability', 'prediction'])

for seq_record in tqdm(SeqIO.parse(fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = seq_record.seq
    
    positive_predicted = []
    negative_predicted = []
    
    # extract protT5 for full sequence and store in temporary dataframe 
    pt5_all = get_protT5_features(sequence)
    
    # generate embedding features and window for each amino acid in sequence
    for index, amino_acid in enumerate(sequence):
        
        # check if AA is 'S' or 'T'
        if amino_acid in ['S', 'T']:
            
            site = index + 1
            
            # get ProtT5 features extracted above
            X_test_pt5 = pt5_all[index]
            
            # load model
            model = load_model('models/ANN_Final_Model.h5')
                        
            y_pred = model.predict(np.array(X_test_pt5.reshape(1,1024)), verbose = 0)[0][0]
            
            # append results to results_df
            results_df.loc[len(results_df)] = [prot_id, site, amino_acid, y_pred, int(y_pred>0.5)]

# Export results 
print('Saving results ...')
results_df.to_csv("output/results.csv", index = False)
print('Results saved to output/results.csv')
