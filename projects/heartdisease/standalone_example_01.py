"""
####### Instructions #######
python standalone_example_01.py


"""
import os, argparse
import numpy as np
import joblib
import torch
import torch.nn as nn
from edattr.setup_interface2 import StandardClassifierS2

NUMERICAL_FEATURES = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
TOKEN_FEATURES =['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma'] # Let's exclude the following 'Sex', 'Race' , 'SkinCancer' , 'KidneyDisease'
TARGET_LABEL_NAME = 'HeartDisease'

TARGET_LABEL_DICTIONARY = { 'No': 0, 'Yes': 1,}
C = len(TARGET_LABEL_DICTIONARY) 

####### Dataset ####### 
# Don't put this guy inside main(), it will cause pickling error
from edattr.setup_interface1 import DatasetSingleClassifierCSV
from edattr.setup_interface2 import DatasetStandardClassifierCSVTypeS2
class HeartDisease_dataset(DatasetStandardClassifierCSVTypeS2):
    def __init__(self, setupTypeS2, split):
        super(HeartDisease_dataset, self).__init__(setupTypeS2, split)

    def __getitem__(self, i):
        idx = self.indices[i] # raw index, straight out of the csv file

        tokens = self.df[self.TOKEN_FEATURES].loc[idx].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[idx].to_numpy()
        x = np.concatenate((tokens, numerics))
        
        ######### !!Here!! #########
        # This is the only part we need to override in this example
        y0 = TARGET_LABEL_DICTIONARY[self.df_target[idx]] 
        ############################        
        return idx, x,y0   
          


def main(**kwargs):

    raise RuntimeError("DO NOT RUN EXPERIMENT WITH THIS FIRST. At a later stage you will encounter pickling error!\nIn progress, we are debugging it...")

    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    DATA_CACHE_DIR = DIRS['DATA_CACHE_DIR']
    if not os.path.exists(DATA_CACHE_DIR):
        print('Please prepare the data first. Run --mode preprocess_normalize.')
        exit()

    cache = joblib.load(DATA_CACHE_DIR)
    dict_leng = len(cache['word_to_ix'])

    config = {
        'n_epochs': 4,
        'learning_rate': 0.2, 
        'batch_size': 32,

        # train/val/test
        'val_fraction':0.01,
        'test_fraction': 0.01,
        'early_stopping':{
            'min_train_iters': 4096,
            'val_every_n_iters': 256,
            'metrics_target': {'acc':0.8, 'recall':0.8},

            ####### for fast debugging #######
            # 'min_train_iters': 256,
            # 'val_every_n_iters': 256,
            # 'metrics_target': {'f1':0.2},            
        },

        # metric_types: any/all of acc, recall, precision, f1
        # warning! for each extra metric type, one new model is saved (memory consideration)
        'metric_types': ['acc','recall'], 

        # models. Different models may require different input
        'model': 'TransformerEmb',
        'layers': {'C':C, 'dim_ff':128}, # need to be defined 
        'tf_conf': {'nhead':4, 'n_enc':2, 'nD': 11,}, # for model = TransformerEmb

        # endorsed attributions
        # RTCS: Reduced Training Class Size threshold
        'RTCS_threshold' : 0.05, # 2560, 
        'RTCS_mode': 'fraction', # 'absolute',
        
        'endorsement_mode': 'shap-lime-top2',
        'eec_modes': ['type-a', 'type-b'],

        # after reduced training set, we can train on larger no. of epochs
        'eec_n_epochs': 64,

        'TOKEN_FEATURES': TOKEN_FEATURES,
        'NUMERICAL_FEATURES': NUMERICAL_FEATURES,
        'TARGET_LABEL_NAME': TARGET_LABEL_NAME,
        'dict_leng': dict_leng, # size of vocabulary 
    }

    # Let's be a little bit object oriented (strictly speaking, not necessary though)
    class classifierHeartDisease(StandardClassifierS2):
        def __init__(self, DIRS, **kwargs):
            self.TARGET_LABEL_NAME = TARGET_LABEL_NAME
            self.TOKEN_FEATURES = TOKEN_FEATURES
            self.NUMERICAL_FEATURES =  NUMERICAL_FEATURES
            # TARGET_LABEL_NAME needs to be set before initiating parent class
            super(classifierHeartDisease, self).__init__(DIRS, **kwargs)

            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.,17.])) # define your loss function here


        def get_lr_lambda(self, epoch):
            # just some default function for learning rate scheduler
            return 0.5 ** epoch

        def set_config(self):
            DATA_CACHE_DIR = self.DIRS['DATA_CACHE_DIR']
            self.config = config

        def set_dataset_object(self):
            self.eec_dataset_object = DatasetSingleClassifierCSV 

            # we're overriding this dataset_object only (see setup_interface1.py)
            self.dataset_object = HeartDisease_dataset        

        def init_new_components(self, **kwargs):
            # let's define a different optimizer for fun
            import torch.optim as optim

            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], betas=(0.5,0.9))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.get_lr_lambda)
            
            components = {'optimizer': optimizer,'scheduler': scheduler,}
            return components

    # imported from setup.py
    cf = classifierHeartDisease(DIRS, **kwargs)

    from edattr.endorse import StandardClassifierEndorsementVis
    cfev = StandardClassifierEndorsementVis(DIRS,**kwargs)    

    TOGGLE = kwargs['toggle']
    if TOGGLE == '0':
        cf.log_model_number_of_params(**kwargs); exit()   
    if TOGGLE is None: TOGGLE = '1111'

    if TOGGLE[0] == '1':
        cf.train_val_test(verbose=100)
        cf.visualize_output()
    if TOGGLE[1] == '1':
        cf.endorse_selected_models(**kwargs)
        cfev.visualize_endorsement_selected_models(**kwargs)  
    if TOGGLE[2] == '1':        
        cf.eec_partition_selected_models(**kwargs)   
        cf.eec_selected_models(**kwargs)   
    if TOGGLE[3] == '1':
        cf.post_eec_train_val_test(**kwargs)
        cf.visualize_post_eec_output(**kwargs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('-m','--mode', default=None, type=str, help=None)  
    parser.add_argument('--DEV_ITER', default=0, type=int, help=None)
    parser.add_argument('--toggle', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary        
    kwargs.update({
        'DATA_FILE_NAME': 'heart_2020_cleaned.csv',
        'full_projectname': 'heartdisease', 
        'prefix': 'branch',
        'feature_mode':'Token+Num',
        'DIRECTORY_MODE': 'singlefile',
        'label': 'heartdisease_standard_transformer_1110-0',
        'WORKSPACE_DIR': None,
        'DATA_DIR': None, 
        'DATA_PROCESSING_DIR': None,
        })    

    main(**kwargs)

