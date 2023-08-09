import joblib
import torch.nn as nn
from edattr.setup_interface2 import kFoldClassifierK2

TOKEN_FEATURES = ['gender', 'smoking']
NUMERICAL_FEATURES = ['score1', 'score2']
TARGET_LABEL_NAME = 'target'

from toydata2 import mt
# mt is the data generator object. 
C = len(mt.columns_setting[TARGET_LABEL_NAME]['choice'])

# Let's define our default config
def myConfig(**kwargs):
    # default template for config. Override it with kwargs
    config = {
        'kfold': 5,
        'n_epochs': 32,
        'learning_rate': 0.0005, 
        'batch_size': 16,    

        # metric_types: any/all of acc, recall, precision, f1
        # warning! for each extra metric type, one new model is saved (memory consideration)
        'metric_types': ['acc','f1'],  # ['recall', 'precision'],

        # models. Different models may require different input
        'model': 'MLPEmb',
        'layers': None, # need to be defined 
        'tf_conf': {}, # for model = TransformerEmb

        'endorsement_mode': 'shap-lime-top2',
        'eec_modes': ['type-a', 'type-b'],

        'TOKEN_FEATURES': TOKEN_FEATURES,
        'NUMERICAL_FEATURES': NUMERICAL_FEATURES,
        'TARGET_LABEL_NAME': TARGET_LABEL_NAME,
        'dict_leng': None, # size of vocabulary 

        # others
        'perturb_n': 64,
    }

    for kwarg, value in config.items():
        if kwarg in kwargs:
            config[kwarg] = kwargs[kwarg]

    print('config:')
    for k,v in config.items():
        print(f'  {k}:{v}')
    return config

def get_config_by_label(**kwargs):
    try:
        suffix, repetition = kwargs['label'].split('-')
    except:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError(f"Please use the following format {kwargs['label']}-0")

    DATA_CACHE_DIR = kwargs['DATA_CACHE_DIR']
    cache = joblib.load(DATA_CACHE_DIR)
    dict_leng = len(cache['word_to_ix'])

    if suffix == 'toy2_kfold_mlp_0000':
        layers = {'nD': 27, 'encoder_out_d': 37, 'fc': [37,C], }
        return myConfig(model='MLPEmb', layers=layers, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_mlp_0001':
        layers = {'nD': 37, 'encoder_out_d': 27, 'fc': [57, 37, 37, C], }
        return myConfig(model='MLPEmb', layers=layers, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_mlp_0002':
        layers = {'nD': 37, 'encoder_out_d': 37, 'fc': [57, 57, 37, C], }
        return myConfig(model='MLPEmb', layers=layers, dict_leng=dict_leng, **kwargs)  
    elif suffix == 'toy2_kfold_resnet_0000': 
        from edattr.model import make_intermediate_layer_settings_eResNetEmb    
        # original resnets:
        # planes=[64,128,256,512]
        # n_blocks=[2,2,2,2] (resnet18), [3,4,6,3] (resnet34)
        # strides=[1,2,2,2]
        iL_settings, emb_setting = make_intermediate_layer_settings_eResNetEmb(
            planes=[1, 2], n_blocks=[1,1], strides=[1,2], nD=5, encoder_out_d=7
        )        
        # C: no. of classes
        layers = {'iL_settings': iL_settings, 'emb_setting':emb_setting, 'C': C,}
        return myConfig(model='ResNetEmb', layers=layers, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_resnet_0001': 
        from edattr.model import make_intermediate_layer_settings_eResNetEmb   
        iL_settings, emb_setting = make_intermediate_layer_settings_eResNetEmb(
            planes=[4, 8], n_blocks=[1,2], strides=[1,2], nD=5, encoder_out_d=7
        )        
        # C: no. of classes
        layers = {'iL_settings': iL_settings, 'emb_setting':emb_setting, 'C': C,}
        return myConfig(model='ResNetEmb', layers=layers, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_resnet_0002': 
        from edattr.model import make_intermediate_layer_settings_eResNetEmb           
        iL_settings, emb_setting = make_intermediate_layer_settings_eResNetEmb(
            planes=[4, 8], n_blocks=[2,2], strides=[1,2], nD=11, encoder_out_d=17
        )        
        # C: no. of classes
        layers = {'iL_settings': iL_settings, 'emb_setting':emb_setting, 'C': C,}
        return myConfig(model='ResNetEmb', layers=layers, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_transformer_0000':
        tf_conf = {'nhead':4, 'n_enc':1, 'nD': 12,}
        layers = {'C':C, 'dim_ff':64}
        return myConfig(model='TransformerEmb', layers=layers, tf_conf=tf_conf, dict_leng=dict_leng, **kwargs)
    elif suffix == 'toy2_kfold_transformer_0001':
        tf_conf = {'nhead':4, 'n_enc':2, 'nD': 12,}
        layers = {'C':C, 'dim_ff':80}
        return myConfig(model='TransformerEmb', layers=layers, tf_conf=tf_conf, dict_leng=dict_leng, **kwargs)        
    elif suffix == 'toy2_kfold_transformer_0002':
        tf_conf = {'nhead':4, 'n_enc':2, 'nD': 17,}
        layers = {'C':C, 'dim_ff':96}
        return myConfig(model='TransformerEmb', layers=layers, tf_conf=tf_conf, dict_leng=dict_leng, **kwargs)        
    else:
        from edattr.factory import clean_up_directory
        clean_up_directory(**kwargs)  # let's do a bit of cleanup
        raise NotImplementedError('Label not recognized?')

class kFold_Toy2(kFoldClassifierK2):
    def __init__(self, DIRS, **kwargs):
        self.TARGET_LABEL_NAME = TARGET_LABEL_NAME
        self.TOKEN_FEATURES = TOKEN_FEATURES
        self.NUMERICAL_FEATURES =  NUMERICAL_FEATURES
        # TARGET_LABEL_NAME needs to be set before initiating parent class
        super(kFold_Toy2, self).__init__(DIRS, **kwargs)

        self.criterion = nn.CrossEntropyLoss() # define your loss function here

    def set_config(self):
        DATA_CACHE_DIR = self.DIRS['DATA_CACHE_DIR']
        self.config = get_config_by_label(DATA_CACHE_DIR=DATA_CACHE_DIR, **self.kwargs) # we define our config above

