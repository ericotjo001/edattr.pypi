"""
This script is intended to create the toy data, and then we will put it into 
the default directory (see DATA_DIR)
"""
import os
import numpy as np
import scipy
import pandas as pd
N_DATA = 512
N_CORRUPT = int(0.1*N_DATA)

TARGET_LABEL_NAME = 'targetlabel'
INPUT_DIMENSION = 7
OUTPUT_DIMENSION = 3

def prep_toyexample_dir(**kwargs):
    DATA_DIR = kwargs['DATA_DIR']
    if DATA_DIR is None:
        from edattr.factory import get_home_path
        HOME_ = get_home_path()        
        WORKSPACE_DIR =  os.path.join(HOME_, "Desktop", "edattr.ws")
        os.makedirs(WORKSPACE_DIR,exist_ok=True)

        DATA_FILE_NAME = kwargs['DATA_FILE_NAME']
        DATA_FOLDER_DIR = os.path.join(WORKSPACE_DIR, 'data')
        os.makedirs(DATA_FOLDER_DIR,exist_ok=True)        
        DATA_DIR = os.path.join(DATA_FOLDER_DIR, DATA_FILE_NAME)
    return DATA_DIR

def hadamard_matrix(n, nearest=False):
    # sometimes called walsh matrix

    # sylvester construction
    two_power = 2**np.ceil(np.log2(n))
    h = scipy.linalg.hadamard(two_power)
    
    # the full walsh matrix is only applicable to matrices of sizes 2^n
    if nearest: return h
    
    # take the sub-matrix if size n by n
    return h[:n,:n]

def save_toyexample_csv(DATA_DIR):
    n_data = N_DATA
    d_features = INPUT_DIMENSION
    C = OUTPUT_DIMENSION # no. of classes

    h = hadamard_matrix(d_features, nearest=False)
    targetlabels_ = np.random.choice(range(d_features), size=n_data)

    toy_data = {
        f'feature{i}': 10 + 0.1*np.random.normal(0,1,size=(n_data,)) 
            for i in range(d_features)
    }
    df = pd.DataFrame(toy_data) + h[targetlabels_,:]

    randomerror = ['', ' '] + [f'error-{i}' for i in range(5)] 

    corrupt_indices = np.random.choice(range(N_DATA), size=(N_CORRUPT,),replace=False)
    corrupt_features = [f'feature{i}' for i in np.random.choice(range(INPUT_DIMENSION), size=(N_CORRUPT,))]
    for i,f in zip(corrupt_indices,corrupt_features):
        df.loc[i,f] = np.random.choice(randomerror)

    df[TARGET_LABEL_NAME] = targetlabels_%C
    df.to_csv(DATA_DIR, index=False)