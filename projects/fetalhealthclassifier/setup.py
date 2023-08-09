import torch
import torch.nn as nn
from edattr.setup_interface1 import kFoldClassifier


""" TARGET_LABEL_DICTIONARY 
Just a minor peculiarity: the data is labelled 1,2,3.
Since we use 0-based index, we turn it into 0,1,2 respectively.
"""
TARGET_LABEL_DICTIONARY = { 
    1: 0, # Normal
    2: 1, # Suspect
    3: 2  # Pathological
}
TARGET_LABEL_NAME = 'fetal_health' 

INPUT_DIMENSION = 21
OUTPUT_DIMENSION = 3

# Let's define our default config
def myConfig(**kwargs):
    # default template for config. Override it with kwargs
    config = {
        'kfold': 5,
        'n_epochs': 128,
        'learning_rate': 0.01,
        'batch_size': 32,    

        # metric_types: any/all of acc, recall, precision, f1
        # warning! for each extra metric type, one new model is saved (memory consideration)
        'metric_types': ['acc','recall'], 

        # models. Different models may require different input
        'model': 'MLP',
        'layers': [], # for model = MLP, ResNet or Transformer
        'resnet_conf': {}, # for model=ResNet
        'tf_conf': {}, # for model = Transformer

        'endorsement_mode': 'shap-lime-top2',
        'eec_modes': ['type-a', 'type-b'],
    }

    for kwarg, value in config.items():
        if kwarg in kwargs:
            config[kwarg] = kwargs[kwarg]

    print('config:')
    for k,v in config.items():
        print(f'  {k}:{v}')
    return config

# This is just a handy function to help us vary our config for experimentation
def get_config_by_label(**kwargs):
    # We arrange our projects according to "label" name, 
    #   like toy_fhc_kfold_mlp_0000-0, fhc_kfold_mlp_0000-1 etc
    # 0,1,,...,m,... at the end are labels for m-th experiment with the same setup. 
    din, dout = INPUT_DIMENSION, OUTPUT_DIMENSION

    try:
        suffix, repetition = kwargs['label'].split('-')
    except:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError(f"Please use the following format {kwargs['label']}-0")
        
    if suffix == 'fhc_kfold_mlp_0000':
        return myConfig(model='MLP', layers=[din, 7*din, dout], **kwargs)
    elif suffix == 'fhc_kfold_mlp_0001':
        return myConfig(model='MLP', layers=[din, 7*din, din, dout], **kwargs)
    elif suffix == 'fhc_kfold_mlp_0002':
        return myConfig(model='MLP', layers=[din, 7*din, 3*din, dout], **kwargs)
    elif suffix == 'fhc_kfold_resnet_0000':
        cf = {'planes':[5,8], 'n_blocks':[1,1]}
        return myConfig(model='ResNet1D', layers=[1, dout],resnet_conf=cf, input_type='single_flat_channel', **kwargs)  
    elif suffix == 'fhc_kfold_resnet_0001':    
        cf = {'planes':[5,8], 'n_blocks':[2,2]}
        return myConfig(model='ResNet1D', layers=[1, dout],resnet_conf=cf, input_type='single_flat_channel', **kwargs)   
    elif suffix == 'fhc_kfold_resnet_0002':    
        cf = {'planes':[6,8], 'n_blocks':[3,4,3]}
        return myConfig(model='ResNet1D', layers=[1, dout],resnet_conf=cf, input_type='single_flat_channel', **kwargs)                
    elif suffix == 'fhc_kfold_transformer_0000':
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':2, 'n_enc':1, 'dim_ff': 32}, **kwargs)
    elif suffix == 'fhc_kfold_transformer_0001':   
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':4, 'n_enc':1, 'dim_ff': 64}, **kwargs) 
    elif suffix == 'fhc_kfold_transformer_0002':
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':4, 'n_enc':2, 'dim_ff': 64}, **kwargs)                  
    else:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError('Label not recognized?')


####### Dataset ####### 
# What's happening here? See kFold_FHC set_dataset_object()
from edattr.setup_interface1 import DatasetKFoldClassifierCSV, DatasetSingleClassifierCSV
class FHC_kfold_dataset(DatasetKFoldClassifierCSV):
    def __init__(self, setupTypeK1, k, split):
        super(FHC_kfold_dataset, self).__init__(setupTypeK1, k, split)

    def __getitem__(self, i):
        # "indices" is a variable introduced by our kfold setup. 
        idx = self.indices[i] # raw index, straight out of the csv file
        x = self.df[idx] # some processing already applied (e.g. normalized with yeo-johnson)
        
        ######### !!Here!! #########
        # This is the only part we need to override in this example
        y0 = TARGET_LABEL_DICTIONARY[int(self.df_target[idx])] 
        ############################        
        return idx, x,y0        
        
# Let's be a little bit object oriented (strictly speaking, not necessary though)
class kFold_FHC(kFoldClassifier):
    def __init__(self, DIRS, **kwargs):
        self.TARGET_LABEL_NAME = TARGET_LABEL_NAME
        # TARGET_LABEL_NAME needs to be set before initiating parent class
        super(kFold_FHC, self).__init__(DIRS, **kwargs)
                
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.,10.,10.])) # define your loss function here

    def set_config(self):
        self.config = get_config_by_label(**self.kwargs) # we define our config above

    def set_dataset_object(self):
        """ fetalhealthclassifier data is stored with labels 1,2,3 rather than 0,1,2 
        i.e it's not in standard pytorch format. Let's replace the dataset objects with
        the ones compatible with fetalhealthclassifier data
        """
        self.eec_dataset_object = DatasetSingleClassifierCSV 

        # we're overriding this dataset_object only (see setup_interface1.py)
        self.dataset_object = FHC_kfold_dataset
