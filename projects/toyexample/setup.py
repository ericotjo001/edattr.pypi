import torch.nn as nn
from edattr.setup_interface1 import kFoldClassifier

from toydata import TARGET_LABEL_NAME, INPUT_DIMENSION, OUTPUT_DIMENSION, N_DATA

# Let's define our default config
def myConfig(**kwargs):
    # default template for config. Override it with kwargs
    config = {
        # !!important!! if learning_rate=0.01, batch_size=32, it is very easy to attain perfect scores. But let's slow down the training so that we get to observe the variation in endorsement results
        'kfold': 5,
        'n_epochs': 64,
        'learning_rate': 0.0001, 
        'batch_size': 8,    

        # metric_types: any/all of acc, recall, precision, f1
        # warning! for each extra metric type, one new model is saved (memory consideration)
        'metric_types': ['acc','f1'], 

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
    #   like toy_kfold_mlp_0001-0, toy_kfold_mlp_0001-1 etc
    # 0,1,,...,m,... at the end are labels for m-th experiment with the same setup.
    din, dout = INPUT_DIMENSION, OUTPUT_DIMENSION

    try:
        suffix, repetition = kwargs['label'].split('-')
    except:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError(f"Please use the following format {kwargs['label']}-0")
        
    if suffix == 'toy_kfold_mlp_0000':
        return myConfig(model='MLP', layers=[din, 7*din, dout], **kwargs)
    elif suffix == 'toy_kfold_mlp_0001':
        return myConfig(model='MLP', layers=[din, 7*din, 7*din, dout], **kwargs)
    elif suffix == 'toy_kfold_mlp_0002':
        return myConfig(model='MLP', layers=[din, 7*din, 14*din, dout], **kwargs)
    elif suffix == 'toy_kfold_resnet_0000':
        cf = {'planes':[2,3], 'n_blocks':[1,1]}
        return myConfig(model='ResNet1D', layers=[1, dout], resnet_conf=cf, **kwargs)       
    elif suffix == 'toy_kfold_resnet_0001':
        cf = {'planes':[4,5], 'n_blocks':[1,1]}
        return myConfig(model='ResNet1D', layers=[1, dout], resnet_conf=cf, **kwargs)       
    elif suffix == 'toy_kfold_resnet_0002':
        cf = {'planes':[4,8], 'n_blocks':[2,2]}
        return myConfig(model='ResNet1D', layers=[1, dout], resnet_conf=cf,**kwargs)        
    elif suffix == 'toy_kfold_transformer_0000':
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':2, 'n_enc':1, 'dim_ff': 64}, **kwargs)
    elif suffix == 'toy_kfold_transformer_0001':
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':4, 'n_enc':1, 'dim_ff': 128}, **kwargs)
    elif suffix == 'toy_kfold_transformer_0002':
        return myConfig(model= 'Transformer', layers=[din, dout], tf_conf={'nhead':4, 'n_enc':2, 'dim_ff': 128}, **kwargs)        
    else:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError('Label not recognized?')

# Let's be a little bit object oriented (strictly speaking, not necessary though)
class kFold_Toy(kFoldClassifier):
    def __init__(self, DIRS, **kwargs):
        self.TARGET_LABEL_NAME = TARGET_LABEL_NAME
        # TARGET_LABEL_NAME needs to be set before initiating parent class
        super(kFold_Toy, self).__init__(DIRS, **kwargs)
                
        self.criterion = nn.CrossEntropyLoss() # define your loss function here

    def set_config(self):
        self.config = get_config_by_label(**self.kwargs) # we define our config above
