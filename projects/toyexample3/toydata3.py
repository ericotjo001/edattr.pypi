import os
import numpy as np
import pandas as pd


N_DATA = 1024*16

from edattr.data import MixedTypeDF
# let's create some synthetic data
mt = MixedTypeDF(nrows=N_DATA, 
    TARGET_CLASSES=[0,1,2], 
    TARGET_CLASSES_PROBABILITIES= [0.6,0.3,0.1], 
    columns_setting=None)

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

def save_toyexample_csv(DATA_DIR):
    df = mt.get_mixed_type_df_random_dataframe() 
    df.to_csv(DATA_DIR, index=False)
