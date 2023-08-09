"""
Setup template is intended to help new users to quick start the project.
We abstract more and more details in the template UP the level, so that, hopefully,
at the user implementation level (highest level), users don't need to care too much about the details

CONVENTION. We will make use of the following names to name template files from lower to higher levels:
setup_template.py (lowest level template) 
setup_interface1.py
setup_interface2.py
setup_interface3.py
...
setup.py (highest level, to be defined by users)

Remark: 
1. we shouldn't have too many setup_interfaces. Otherwise, our code design is probably bad.
2. Ideally, higher level templates use templates from lower levels. 
"""

from .utils import *
from .data import kFoldPrep, kSplitDataset, StandardPrep, StandardDataset
from torch.utils.data import Dataset

def init_new_template_model(**config):
    model = config['model']
    if model == 'MLP':
        from .model import MLP
        # For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)
        return MLP(config['layers'])
    
    elif model == 'ResNet1D':
        from .model import eResNet, make_intermediate_layer_settings_eResNet
        # For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)

        cf = config['resnet_conf']
        planes = cf['planes']
        n_blocks = cf['n_blocks']
        s = make_intermediate_layer_settings_eResNet(
            planes=planes, n_blocks=n_blocks, strides=[1,2], 
        ) # see projects/resnets.py for more comments               

        L = config['layers']
        return eResNet(1, L[-1], s, dim=1,input_type='single_flat_channel')

    elif model == 'Transformer':
        tfc = config['tf_conf']
        nhead, n_enc, dim_ff = tfc['nhead'], tfc['n_enc'], tfc['dim_ff']
        L = config['layers']

        from .model import  eTFClassifier
        # For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)
        # Transformer for classifier 

        INPUT_DIMENSION, OUT_DIMENSION = L[0], L[-1]
        return eTFClassifier(INPUT_DIMENSION, OUT_DIMENSION, nhead=nhead, n_enc=n_enc, dim_ff=dim_ff)

    elif model == 'MLPEmb':
        from .model import MLPEmb
        # For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)
        return MLPEmb(config['layers'], config['TOKEN_FEATURES'], config['NUMERICAL_FEATURES'], config['dict_leng'])

    elif model == 'ResNetEmb':
        from .model import eResNetEmb1D
        # For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)

        conf = config
        L = config['layers']
        return eResNetEmb1D(conf['TOKEN_FEATURES'], conf['NUMERICAL_FEATURES'], conf['dict_leng'], L['iL_settings'], L['emb_setting'], L['C'])

    elif model == 'TransformerEmb':
        from .model import eTFClassifierEmb
        # For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)   
        conf = config
        tf_conf = config['tf_conf']
        return eTFClassifierEmb(conf['TOKEN_FEATURES'], conf['NUMERICAL_FEATURES'], conf['dict_leng'], conf['layers']['C'], dim_ff=conf['layers']['dim_ff'], nD=tf_conf['nD'], nhead=tf_conf['nhead'], n_enc=tf_conf['n_enc'],)        

    else:
        print(model)
        raise NotImplementedError('Model not recognized?')    


#############################################
#                   TypeR1
#############################################
""" 
TypeR1: Raw dataframe. Data is from CSV file.
CSV file can be loaded as pandas dataframe that has 
1. many columns for features and 
2. one column for target (TARGET_LABEL_NAME). No restriction on data type of target.

Data is loaded without any preprocessing.
Usage: for processed data stored CSV files. For example, in our EEC process, preprocessed data are selected in part through KMeans algorithm and stored in CSV
"""        

class DataSetupTypeR1():
    def __init__(self, RAW_DF_DIR, TARGET_LABEL_NAME):
        super(DataSetupTypeR1, self).__init__() # nothing here btw

        self.RAW_DF_DIR = RAW_DF_DIR
        self.TARGET_LABEL_NAME = TARGET_LABEL_NAME        
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.RAW_DF_DIR, index_col=False)   
        features = [feature for feature in df if not feature == self.TARGET_LABEL_NAME]
        df_features = df.loc[:, features]      

        self.df = df_features.to_numpy()
        self.df_target = df.loc[:, self.TARGET_LABEL_NAME]                

class DatasetTypeR1(Dataset):
    def __init__(self, setupTypeR1):
        # setupTypeR1 is DataSetupTypeR1 (see above)
        super(DatasetTypeR1, self).__init__() # nothing here btw

        self.df = setupTypeR1.df
        self.df_target = setupTypeR1.df_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # overwrite this function on case by case basis
        x = self.df[i]
        y0 = self.df_target[i]
        return x, y0            


#############################################
#                   TypeK1
#############################################
"""   
!! See data.py "Data CONVENTION" to distinguish between kFoldPrep and Dataset object.

Data is from CSV file. Data will be indexed for k-fold validation.
CSV file can be loaded as pandas dataframe that has 
1. many columns, each contains ONLY numbers which can be normalized
2. no missing or invalid data (data cleaning is done)
3. one column for target (TARGET_LABEL_NAME). No restriction on data type of target.

Data Frame Type: NAIVE NUMERICAL DATAFRAME (refer to data.py for more details)

Features are transformed with sklearn pipeline or anything similar that uses .transform()
The pipeline has been saved as a cache in DATA_CACHE_DIR

Since this performs kfold validation, it's more suitable for smaller dataset:
  repeated training and validation of large models is computationally costly
"""

class DataSetupTypeK1(kFoldPrep):
    # A kFoldPrep Object 
    def __init__(self, DIRS, **kwargs):
        # Parent classes' initiations include
        # self.indices where
        # indices[k] = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        self.TARGET_LABEL_NAME = kwargs['TARGET_LABEL_NAME']

        # temporary setup of DIRS because we wanna quickly load the data first 
        # We will load the full dir via kFoldPrep init
        self.DIRS = {
            'DATA_DIR': DIRS['DATA_DIR'],
            'DATA_CACHE_DIR': DIRS['DATA_CACHE_DIR']} 

        self.load_data() # load data first

        # This is the kFoldPrep init (super class)
        super(DataSetupTypeK1, self).__init__(DIRS, **kwargs)

    def load_data(self):
        from edattr.data import replace_invalid_cell_with_blank

        # !! Important !! This should mirror dfPP.process_dataframe() in data.py 
        df = pd.read_csv(self.DIRS['DATA_DIR'], index_col=False)     

        features = [feature for feature in df if not feature == self.TARGET_LABEL_NAME]
        df_features = df.loc[:, features]    

        df_features = df_features.applymap(replace_invalid_cell_with_blank, ftype='numeric')   
        
        cache = joblib.load(self.DIRS['DATA_CACHE_DIR'])
        pipe = cache['feature_transform_pipeline']

        self.df = pipe.transform(df_features) # = df_ 
        self.df_target = df.loc[:, self.TARGET_LABEL_NAME]        

    def get_data_size(self):
        return len(self.df)

class DatasetTypeK1(kSplitDataset):
    # Note: kSplitDataset is directly descended from a pytorch Dataset object (from torch.utils.data import Dataset). See data.py
    def __init__(self, setupTypeK1, k, split):
        # setupTypeK1 is DataSetupTypeK1() (see above)
        super(DatasetTypeK1, self).__init__(setupTypeK1, k, split)
        self.df = setupTypeK1.df 
        self.df_target = setupTypeK1.df_target

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # overwrite this function on case by case basis
        idx = self.indices[i] # raw index from the CSV file
        x = self.df[idx]
        y0 = self.df_target[idx]
        return idx, x, y0            


#############################################
#                   TypeK2
#############################################
"""   
!! See data.py "Data CONVENTION" to distinguish between Setup Object (like kFoldPrep) and Dataset object.

Data is from CSV file. Data will be indexed for k-fold validation.
CSV file can be loaded as pandas dataframe that has 
1. columns of data that can be converted to either (1) string (2) float.
2. one column for target (TARGET_LABEL_NAME). No restriction on data type of target.

Data Frame Type: TokenAndFloat DATAFRAME (refer to data.py for more details)

# Features are transformed with sklearn pipeline or anything similar that uses .transform()
# The pipeline has been saved as a cache in DATA_CACHE_DIR

Since this performs kfold validation, it's more suitable for smaller dataset:
  repeated training and validation of large models is computationally costly
"""


class DataSetupTypeK2(kFoldPrep):
    def __init__(self, DIRS, **kwargs):
        self.TARGET_LABEL_NAME = kwargs['TARGET_LABEL_NAME']
        self.TOKEN_FEATURES = kwargs['TOKEN_FEATURES']
        self.NUMERICAL_FEATURES = kwargs['NUMERICAL_FEATURES']

        # temporary setup of DIRS because we wanna quickly load the data first 
        # We will load the full dir via kFoldPrep init
        self.DIRS = {
            'DATA_DIR': DIRS['DATA_DIR'],
            'DATA_CACHE_DIR': DIRS['DATA_CACHE_DIR']} 

        self.load_data() # load data first

        super(DataSetupTypeK2, self).__init__(DIRS, **kwargs)

    def load_data(self):
        df = pd.read_csv(self.DIRS['DATA_DIR'], index_col=False)     

        cache = joblib.load(self.DIRS['DATA_CACHE_DIR'])
        NUMERICAL_FEATURES = cache['NUMERICAL_FEATURES']
        TOKEN_FEATURES = cache['TOKEN_FEATURES']
        word_to_ix = cache['word_to_ix']

        ##### numerical part #####
        pipe = cache['numerical_feature_transform_pipeline']
        df[NUMERICAL_FEATURES] = pipe.transform(df[NUMERICAL_FEATURES])

        ##### tokens part #####
        def word_to_ix_mapping(x):
            return word_to_ix[x] if x in word_to_ix else 0
        df[TOKEN_FEATURES] = df[TOKEN_FEATURES].applymap(word_to_ix_mapping)        

        self.df = df
        self.df_target = df.loc[:, self.TARGET_LABEL_NAME]        

    def get_data_size(self):
        return len(self.df)

class DatasetTypeK2(kSplitDataset):
    # setupTypeK2 is DataSetupTypeK2() (see above)
    def __init__(self, setupTypeK2, k, split):
        super(DatasetTypeK2, self).__init__(setupTypeK2, k, split)
        self.df = setupTypeK2.df 
        self.df_target = setupTypeK2.df_target         

        self.TARGET_LABEL_NAME = setupTypeK2.TARGET_LABEL_NAME
        self.TOKEN_FEATURES = setupTypeK2.TOKEN_FEATURES
        self.NUMERICAL_FEATURES = setupTypeK2.NUMERICAL_FEATURES

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # overwrite this function on case by case basis
        idx = self.indices[i] # raw index from the CSV file

        tokens = self.df[self.TOKEN_FEATURES].loc[idx].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[idx].to_numpy()
        x = np.concatenate((tokens, numerics)) # convention: token first then numeric <-- the order matters!

        y0 = self.df_target[idx]
        return idx, x, y0            


#############################################
#                   TypeS2
#############################################
"""   
!! See data.py "Data CONVENTION" to distinguish between Setup Object (like kFoldPrep) and Dataset object.

Data is from CSV file. Data will be indexed for train/val/test
CSV file can be loaded as pandas dataframe that has 
1. columns of data that can be converted to either (1) string (2) float.
2. one column for target (TARGET_LABEL_NAME). No restriction on data type of target.

Data Frame Type: TokenAndFloat DATAFRAME (refer to data.py for more details)

# Features are transformed with sklearn pipeline or anything similar that uses .transform()
# The pipeline has been saved as a cache in DATA_CACHE_DIR

This is suitable for very large dataset. For example, if you have 100k (one hundred thousand) data points, you can draw 2k for validation and 2k for testing. You can always choose more data points for val/test if you have enough resources
"""

class DataSetupTypeS2(StandardPrep):
    def __init__(self, DIRS, **kwargs):
        self.TARGET_LABEL_NAME = kwargs['TARGET_LABEL_NAME']
        self.TOKEN_FEATURES = kwargs['TOKEN_FEATURES']
        self.NUMERICAL_FEATURES = kwargs['NUMERICAL_FEATURES']

        # temporary setup of DIRS because we wanna quickly load the data first 
        self.DIRS = {
            'DATA_DIR': DIRS['DATA_DIR'],
            'DATA_CACHE_DIR': DIRS['DATA_CACHE_DIR']} 

        self.load_data() # load data first

        super(DataSetupTypeS2, self).__init__(DIRS, **kwargs)

    def load_data(self):
        df = pd.read_csv(self.DIRS['DATA_DIR'], index_col=False)      

        cache = joblib.load(self.DIRS['DATA_CACHE_DIR'])
        NUMERICAL_FEATURES = cache['NUMERICAL_FEATURES']
        TOKEN_FEATURES = cache['TOKEN_FEATURES']
        word_to_ix = cache['word_to_ix']

        ##### numerical part #####
        from edattr.data import replace_invalid_cell_with_blank
        pipe = cache['numerical_feature_transform_pipeline']
        df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].applymap(replace_invalid_cell_with_blank, ftype='numeric').convert_dtypes()        
        df[NUMERICAL_FEATURES] = pipe.transform(df[NUMERICAL_FEATURES])

        ##### tokens part #####
        def word_to_ix_mapping(x):
            return word_to_ix[x] if x in word_to_ix else 0
        df[TOKEN_FEATURES] = df[TOKEN_FEATURES].applymap(word_to_ix_mapping)        

        self.df = df
        self.df_target = df.loc[:, self.TARGET_LABEL_NAME]        

    def get_data_size(self):
        return len(self.df)    

    def save_new_indices(self, n, DATA_STANDARD_INDICES_DIR, 
        val_fraction=0.02, test_fraction=0.02, **kwargs):
        # Overwrite the default save_new_indices() function available in StandardPrep 
        
        from .data import get_standard_weighted_indices
        self.indices = get_standard_weighted_indices(n, self.df_target, 
            shuffle=self.shuffle, 
            val_fraction=val_fraction, 
            test_fraction=test_fraction,
            NOTE_DIR=DATA_STANDARD_INDICES_DIR + '.txt',
        )
        joblib.dump(self.indices, DATA_STANDARD_INDICES_DIR)        


class DatasetTypeS2(StandardDataset):
    # setupTypeS2 is DataSetupTypeS2() (see above)
    def __init__(self, setupTypeS2, split):
        super(DatasetTypeS2, self).__init__(setupTypeS2, split)
        self.df = setupTypeS2.df 
        self.df_target = setupTypeS2.df_target         

        self.TARGET_LABEL_NAME = setupTypeS2.TARGET_LABEL_NAME
        self.TOKEN_FEATURES = setupTypeS2.TOKEN_FEATURES
        self.NUMERICAL_FEATURES = setupTypeS2.NUMERICAL_FEATURES

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # overwrite this function on case by case basis
        idx = self.indices[i] # raw index from the CSV file

        tokens = self.df[self.TOKEN_FEATURES].loc[idx].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[idx].to_numpy()
        x = np.concatenate((tokens, numerics)) # convention: token first then numeric <-- the order matters!

        y0 = self.df_target[idx]
        return idx, x, y0            
