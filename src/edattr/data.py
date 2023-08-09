"""
******* Data CONVENTION *******
Distinguish between "Setup object" (like kFoldPrep) and "Dataset object" (like kSplitDataset).
1. Dataset object: this is derived from torch.utils.data import Dataset
   It will be the very object that gets fed into DataLoader (from torch.utils.data DataLoader)
   for randomization and batching
2. Setup object: this is a generic object to prepare data. For example, in kFoldPrep 
   for k-fold training, validation and testing, the job is to prepare the indices. It will
   include data loading function. This setup object will be fed into a Dataset object for
   initialization.


******* Data Frame Types *******
1. NAIVE NUMERICAL DATAFRAME
    Super simple, naive version with the following assumptions:
    1. All other columns in df consist of numbers that can be normalized
    ** df can have some invalid values or empty cells (updated)

2. TokenAndFloat DATAFRAME
In this framework, 
    a) strings will be tokens for embedding in the model
    b) Ordinals and bools will be converted to string tokens as well (rather than one-hot encoding etc)
    c) Numbers that can be meaningfully ordered like real numbers (in contrast to bool and ordinals) are treated as floating point numbers. 

""" 

from .utils import *
from .decorator import *
from torch.utils.data import Dataset

ASSUMPTION_VIOLATED = "One of the assumptions are violated."
RESOLVE_TBD_MSG = """Your next step is to set TOKEN_FEATURES and NUMERICAL_FEATURES in your script. This tells the algorithm what is the format of each column of data, and this affects how it is processed.

* We're not sure about features in _TBD_ (TO BE DETERMINED). Please set them manually, i.e. put them into either NUMERICAL_FEATURES or TOKEN_FEATURES during your setup.

** Some data type in _TBD_ may be obviously numerical to you. We use a simple way to characterize data types that's why we leave the final decision to you. If the data in that column numerical, but there are less than 1024 different values (for example, many data with the same values), to be safe, we consider them undecided. 

*** Finally, watch out for columns of data that contains ID. It might be misclassified as NUMERICAL. Please remove them accordingly.
""" 

####################################
#            Standard
####################################

def get_standard_indices(n, shuffle=False, indices_list=None, val_fraction=0.02, test_fraction=0.02):
    # simple train/val/test. Doesn't care about class imbalance    
    ntrain = int(n*(1 - val_fraction - test_fraction))
    nval = int(n*val_fraction)

    if indices_list is None:
        all_ = list(np.arange(n))
    else:
        all_ = indices_list
        assert(n==len(indices_list))

    train_idx = random.sample(all_, ntrain) # is a list

    remaining_ = np.setdiff1d(all_,train_idx).tolist() # [ x for x in all_ if x not in train_idx] too slow
    val_idx = random.sample(remaining_, nval)
    test_idx = np.setdiff1d(remaining_,val_idx).tolist() # [x for x in remaining_ if x not in val_idx]

    indices = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }
    return indices

def get_standard_weighted_indices(n, df_target, shuffle=False, val_fraction=0.02, test_fraction=0.02, NOTE_DIR=None):
    # df_target: dataframe containing the class/target.
    # simple train/val/test. Will try to balance the dataset according to class proportions
    
    classwise_indices = {c :df_target[df_target==c].index.tolist()  for c in pd.unique(df_target)}

    indices = {'train':[], 'val':[], 'test':[]}
    for c, indices_list in classwise_indices.items():
        indices_ = get_standard_indices(len(indices_list), shuffle=shuffle, indices_list=indices_list, val_fraction=val_fraction, test_fraction=test_fraction)
        for split in indices_:
            indices[split].extend(indices_[split])

    if NOTE_DIR is not None:
        record_classwise_proportion(df_target, classwise_indices, indices, NOTE_DIR)

    return indices    

def record_classwise_proportion(df_target, classwise_indices, indices, NOTE_DIR):
    txt = open(NOTE_DIR, 'w')
    for split in indices:
        txt.write(f'{split}:\n')
        
        df_ = df_target[indices[split]]
        ndf_ = len(df_)
        for c in classwise_indices:
            nc = len(df_[df_==c])
            txt.write(f'  c={c} -> {nc}|{np.round(nc/ndf_*100,2)}%\n')
    txt.close() 

class StandardPrep():
    """ ====== Setup Object for preparing standard train/val/test splits ======
    This object split the data abstractly. We assume that each data entry can be indexed by i = 0,1,2,...,n-1.
    This is more suitable for large dataset. We typically set val and test to be only a fraction of the large dataset
    """
    def __init__(self, DIRS, **kwargs):
        super(StandardPrep, self).__init__()
        self.DIRS = DIRS
        self.kwargs = kwargs
        self.adjust_config_settings()

        self.shuffle = kwargs['shuffle'] # set this to True only during debug

        self.create_or_load_indices(**kwargs)

    def adjust_config_settings(self):
        required_args = []
        optional_args = {
            'new_index': False,
            'verbose': 0,
            'shuffle': True,
        }

        kwargs = self.kwargs
        for arg in required_args:
            if not arg in kwargs:
                raise RuntimeError(f'Missing argument:{arg}')

        for arg, defaultval in optional_args.items():
            if not arg in kwargs:
                self.kwargs[arg] = defaultval        

    def get_data_size(self):
        # Need to know the total number of data points. 
        #   We will then split them into train/val/test.
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    @printfunc    
    def create_or_load_indices(self, **kwargs):
        n = self.get_data_size()
        THIS_DIR = self.DIRS['DATA_STANDARD_INDICES_DIR']

        CREATE_NEW_INDICES = False
        if not os.path.exists(THIS_DIR):  CREATE_NEW_INDICES = True
        if self.kwargs['new_index']:  CREATE_NEW_INDICES = True 

        if CREATE_NEW_INDICES:
            self.save_new_indices(n, THIS_DIR, **kwargs)
            STATUS = f"Saving new train/val/test indices to {THIS_DIR}..."
        else:
            self.indices = joblib.load(THIS_DIR)
            STATUS = f"Loading train/val/test indices from {THIS_DIR}..."
        return STATUS

    def save_new_indices(self, n, DATA_STANDARD_INDICES_DIR, 
        val_fraction=0.02, test_fraction=0.02, **kwargs):
        self.indices = get_standard_indices(n, shuffle=self.shuffle, 
            val_fraction=val_fraction, 
            test_fraction=test_fraction,
        )
        joblib.dump(self.indices, DATA_STANDARD_INDICES_DIR)        

class StandardDataset(Dataset):
    def __init__(self, standPrep, split):
        # standPrep is a StandardPrep object)
        super(StandardDataset, self).__init__()
        self.indices = standPrep.indices[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # the raw index 
        idx = self.indices[i] 

        # the feature vector of the input data. Now it's set to a dummy value
        # During implementation, override x
        # Typical example, x = df[ self.indices[i] ] where df is some dataframe
        x = self.indices[i] 

        # the groundtruth. Now it's just set to a dummy value
        y0 = -1 
        return idx, x, y0

####################################
#             k-Fold
####################################

def get_kfold_indices(n, kfolds=5, shuffle=False):
    from sklearn.model_selection import KFold, train_test_split
    folds = KFold(n_splits=kfolds, shuffle=shuffle ).split(range(n))
    indices = {}
    for k, (trainval_idx, test_idx) in enumerate(folds):
        ntest = int(n/kfolds)
        fraction = ntest/len(trainval_idx) 
        # this fraction is set such that validation and test set have approx. the same size

        train_valtest = train_test_split(trainval_idx, shuffle=shuffle, test_size=fraction)
        train_idx = train_valtest[0]
        val_idx = train_valtest[1]

        indices[k] = {
            'train': train_idx, 'val': val_idx, 'test': test_idx
        }
    return indices


class kFoldPrep():
    """ ====== Setup Object for preparing k fold splits ======
    This object split the data abstractly. We assume that each data entry can be indexed by i = 0,1,2,...,n-1.
    """
    def __init__(self, DIRS, **kwargs):
        super(kFoldPrep, self).__init__()
        self.DIRS = DIRS
        self.kwargs = kwargs
        self.adjust_config_settings()

        self.shuffle = kwargs['shuffle'] # set this to True only during debug

        # Initiate the following:
        # self.indices = {} # k is the k-th fold
        # self.indices[k] = {'train': train_idx, 'val': val_idx, 'test': test_idx}        
        self.create_or_load_kfold_indices(verbose=kwargs['verbose'])

    def adjust_config_settings(self):
        required_args = [
            'kfold', # no of k fold for validations
        ]
        optional_args = {
            'new_index': False,
            'verbose': 0,
            'shuffle': True,
        }

        kwargs = self.kwargs
        for arg in required_args:
            if not arg in kwargs:
                raise RuntimeError(f'Missing argument:{arg}')

        for arg, defaultval in optional_args.items():
            if not arg in kwargs:
                self.kwargs[arg] = defaultval        

    def get_data_size(self):
        # Need to know how many data points are there:
        #   we need to know the total number of entries to split into kfold
        # For example, if you load CSV file into pandas DataFrame, you can use 
        # pandas dataframe method to fetch the no. of rows of the entire data
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    @printfunc    
    def create_or_load_kfold_indices(self, **kwargs):
        n = self.get_data_size()
        THIS_DIR = self.DIRS['DATA_KFOLD_INDICES_DIR']

        CREATE_NEW_KFOLD = False
        if not os.path.exists(THIS_DIR):  CREATE_NEW_KFOLD = True
        if self.kwargs['new_index']:  CREATE_NEW_KFOLD = True 

        if CREATE_NEW_KFOLD:
            self.save_new_indices(n, self.kwargs['kfold'], THIS_DIR)
            STATUS = f"Saving new train/val/test indices to {THIS_DIR}..."
        else:
            self.indices = joblib.load(THIS_DIR)
            STATUS = f"Loading train/val/test indices from {THIS_DIR}..."
        return STATUS

    def save_new_indices(self, n, kfolds, DATA_KFOLD_INDICES_DIR):
        self.indices = get_kfold_indices(n, kfolds=kfolds, shuffle=self.shuffle)
        joblib.dump(self.indices, DATA_KFOLD_INDICES_DIR)

    def get_number_of_folds(self):
        nfold = len(self.indices)
        return nfold


class kSplitDataset(Dataset):
    """ kSplitDataset: k-th fold dataset for a given split (train/val/test)
    This will be the object that is loaded into pytorch DataLoader
    It is designed to split the data safely.

    This is the abstract version. To implement this dataset in practical usage, 
      make sure that your data can be queried with index i like __getitem__
    """
    def __init__(self, kFoldPrep, k, split):
        super(kSplitDataset, self).__init__()
        # kFoldPrep is a kFoldPrep object (of course, it is)        
        self.indices = kFoldPrep.indices[k][split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # the raw index 
        idx = self.indices[i] 

        # the feature vector of the input data. Now it's set to a dummy value
        # During implementation, override x
        # Typical example, x = df[ self.indices[i] ] where df is some dataframe
        x = self.indices[i] 

        # the groundtruth. Now it's just set to a dummy value
        y0 = -1 
        return idx, x, y0


#################################################
#    Preprocessing dataframe from  CSV file
#################################################

def replace_invalid_cell_with_blank(x, ftype='numeric'):
    # ftype: feature type. It's like dtype, but we are only interested in numeric or tokens
    if ftype == 'numeric':
        try:
            float(x)
        except:
            return np.NaN 
    elif ftype == 'token':
        raise NotImplementedError('to-do') # we don't need this actually, since the embedding already does what we need to do.
    else:
        raise NotImplementedError('Nope! ftype is either a numeric or token')
    return x

class dfPP():
    """ dataframe PreProcessor 
    For NAIVE NUMERICAL DATAFRAME (see Data Frame Types at the start of this document)
    """
    def __init__(self, ):
        super(dfPP, self).__init__()

    def process_dataframe(self, df, DATA_CACHE_DIR, TARGET_LABEL_NAME, verbose=0):
        """
        Assumptions:
        1. df has one column that stores the class (target) for classification  
        2. the rest of the columns are numerical
        
        df                 : pandas dataframe
        DATA_CACHE_DIR     : str
        TARGET_LABEL_NAME  : str, name of the column that corresponds to the class (target)

        Warning:
        1. only transform features. TARGET_LABEL columns is left as it is
        """

        features = [feature for feature in df if not feature == TARGET_LABEL_NAME]
        df_features = df.loc[:, features]

        df_features = df_features.applymap(replace_invalid_cell_with_blank, ftype='numeric')

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, PowerTransformer
        from sklearn.impute import KNNImputer
        pipe = Pipeline([
            ('imputer', KNNImputer(n_neighbors=7, weights="distance")),
            ('scaler', StandardScaler()),
            ('pt', PowerTransformer(method='yeo-johnson', standardize=False))
            ]).fit(df_features)

        df_ = pipe.transform(df_features)
        df_processed = numpy_array_to_df(df_, features)        

        cache = {  
            'feature_transform_pipeline': pipe,
            'features': features,
        }
        joblib.dump(cache, DATA_CACHE_DIR)

        processed_ = {'df_processed':df_processed}
        return processed_


class DataVis():
    """
    Assumption: 
    1. assumptions in dfPP process_dataframe() are satisfied

    Warning:
    1. any column that has NaN values will be skipped
    """
    def __init__(self, DATA_VIS_DIR):
        super(DataVis, self).__init__()
        self.DATA_VIS_DIR = DATA_VIS_DIR

        # visualization
        self.nrow, self.ncol = 1,3
        os.makedirs(self.DATA_VIS_DIR, exist_ok=True)

    def vis(self, df, df_processed=None, exclude_target_label=True, TARGET_LABEL_NAME=None,
        verbose=0):
        print('visualizing data...')

        # here, just make sure that we show all that can be shown in the raw df 
        # Remove NaN or broken cells
        df = df.applymap(replace_invalid_cell_with_blank, ftype='numeric').dropna().astype(float)

        exclude_ = []
        if exclude_target_label:
            exclude_.append(TARGET_LABEL_NAME)

        for i,column in enumerate(df.columns):
            plt.figure(figsize=(12,5))
            plt.gcf().add_subplot(self.nrow,self.ncol,1)
            plt.gca().hist(df[column], alpha=0.3, edgecolor='black')
            plt.gca().set_xlabel(column)            
            plt.gcf().add_subplot(self.nrow,self.ncol,2)
            plt.scatter(df[column],df[column],3)

            if not df_processed is None:
                if not column in exclude_: 
                    self.visualize_preprocessed_data(df_processed[column])

            plt.tight_layout()
            IMG_DIR = os.path.join(self.DATA_VIS_DIR, f'{column}.png')
            plt.savefig(IMG_DIR)
            plt.close()

    def vis_mixed_types(self, df, df_processed=None, exclude_target_label=True, TARGET_LABEL_NAME=None, TOKEN_FEATURES=(), word_to_ix={},
        verbose=0):
        print('visualizing mixed types data...')
        ix_to_word = create_ix_to_word(word_to_ix)
        ix_to_word[1] = '-blank-' # for visualization

        exclude_ = []
        if exclude_target_label:
            exclude_.append(TARGET_LABEL_NAME)

        for i,column in enumerate(df.columns):
            plt.figure(figsize=(12,5))
            plt.gcf().add_subplot(self.nrow,self.ncol,1)
            if column in TOKEN_FEATURES:
                plt.gca().hist([ix_to_word[x] for x in df[column]], alpha=0.3, edgecolor='red')

                plt.gca().tick_params(axis='x', rotation=-70)
            else:
                plt.gca().hist(df[column], alpha=0.3, edgecolor='black')
            plt.gca().set_xlabel(column)            
            plt.gcf().add_subplot(self.nrow,self.ncol,2)
            plt.scatter(df[column],df[column],3)

            if not df_processed is None:
                if not column in exclude_: 
                    self.visualize_preprocessed_data(df_processed[column])

            plt.tight_layout()
            IMG_DIR = os.path.join(self.DATA_VIS_DIR, f'{column}.png')
            plt.savefig(IMG_DIR)
            plt.close()            

    def visualize_preprocessed_data(self, dfp_column):
        plt.gcf().add_subplot(self.nrow,self.ncol,3)
        freq, bins, patches = plt.gca().hist( dfp_column, 
            alpha=0.5, color='green', edgecolor='red')
        # just adding numbers to histogram
        # https://stackoverflow.com/questions/63200484/how-to-add-or-annotate-value-labels-or-frequencies-on-a-matplotlib-histogra
        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        n = 0
        toggle=1
        for fr, x, patch in zip(freq, bin_centers, patches):
            height = int(freq[n])
            if height>0 and toggle==1: 
                plt.annotate("{}".format(height),xy = (x, height), xytext = (0,0.2), textcoords = "offset points",ha = 'center', va = 'bottom')
                toggle = 0
            else:
                toggle = 1
            n = n+1

def collect_vocabulary(df, TOKEN_FEATURES, threshold=4):
    # Let's force users to supply TOKEN_FEATURES
    # Only columns included in TOKEN_FEATURES will be considered for vocabulary listing
    # This is mainly intended to prevent floating point values and unique identifiers from being included
    n = len(df)

    k = 2
    word_to_ix = {'_UNK_':0, '': 1}
    for feature in TOKEN_FEATURES:
        # feature: column name

        counttable = df[feature].value_counts()
        for x in counttable[counttable>threshold].to_dict():
            if x in word_to_ix: continue
            word_to_ix[x] = k
            k += 1
        for x in counttable[counttable<=threshold].to_dict():
            word_to_ix[x] = 0 # too few instances, considered unknown _UNK_

    return sort_dictionary_by_values_desc(word_to_ix)

def create_ix_to_word(word_to_ix):
    ix_to_word = [0 for _ in range(len(word_to_ix))]
    for w,ix in word_to_ix.items():
        ix_to_word[ix] = w
    ix_to_word[0] = '_UNK_'
    ix_to_word[1] = ''
    return ix_to_word 

class DataFramePreProcessorTypeA():
    """
    For TokenAndFloat DATAFRAME (see Data Frame Types at the start of this document)
    """
    def __init__(self, ):
        super(DataFramePreProcessorTypeA, self).__init__()

    def process_dataframe(self, df, DATA_CACHE_DIR, 
        TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME,  
        verbose=0):
        TOKEN_FEATURES = list(TOKEN_FEATURES) # just in case
        NUMERICAL_FEATURES = list(NUMERICAL_FEATURES) # just in case

        ##### numerical part #####
        print('handling numerical part...')
        df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].applymap(replace_invalid_cell_with_blank, ftype='numeric').convert_dtypes()

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, PowerTransformer
        from sklearn.impute import KNNImputer
        print('  sklearn pipe.fit in progress...')        
        pipe = Pipeline([
            ('imputer', KNNImputer(n_neighbors=7, weights="distance")),
            ('scaler', StandardScaler()),
            ('pt', PowerTransformer(method='yeo-johnson', standardize=False))
            ], verbose=True).fit(df.loc[:, NUMERICAL_FEATURES])
        
        print('  sklearn pipe.transform in progress...')        
        df[NUMERICAL_FEATURES] = pipe.transform(df[NUMERICAL_FEATURES])

        ##### tokens part #####
        print('handling token part...')
        word_to_ix = collect_vocabulary(df[TOKEN_FEATURES], TOKEN_FEATURES)            
        def word_to_ix_mapping(x):
            return word_to_ix[x] if x in word_to_ix else 0

        print('  token mapping in progress...')
        df[TOKEN_FEATURES] = df[TOKEN_FEATURES].applymap(word_to_ix_mapping)

        ##### exclude the rest #####
        df = df[TOKEN_FEATURES + NUMERICAL_FEATURES]

        cache = {  
            'NUMERICAL_FEATURES': NUMERICAL_FEATURES,
            'numerical_feature_transform_pipeline': pipe,
            'TOKEN_FEATURES': TOKEN_FEATURES,
            'word_to_ix': word_to_ix,
        }
        joblib.dump(cache, DATA_CACHE_DIR)

        processed_ = {
            'df_processed':df, 
            'word_to_ix': word_to_ix,
            }
        return processed_

    def suggest_feature_types(self, df, FEATURE_CACHE_DIR, TARGET_LABEL_NAME=None):
        if os.path.exists(FEATURE_CACHE_DIR): return
        
        suggested_types = dataframe_suggested_types(df,TARGET_LABEL_NAME=TARGET_LABEL_NAME)

        print("======= suggest_feature_types =======")

        txt = open(FEATURE_CACHE_DIR, 'w')
        headline = 'SUGGESTED FEATURES:' 
        print(headline)
        txt.write(headline + '\n')
        for feature_name, suggested_type in suggested_types.items():
            feature_suggestion = f'{"  %-18s"%(str(feature_name))} = {suggested_type}'
            txt.write(feature_suggestion + '\n\n')
            print(feature_suggestion, '\n')
        txt.write('\n\n'+RESOLVE_TBD_MSG)
        print('\n'+RESOLVE_TBD_MSG)
        txt.close()

        print(f'\nThe above message is also stored at {FEATURE_CACHE_DIR} for your reference.')
        print("""\nHello! Now we're exiting the pipeline via "suggest_feature_types" sub-process. This sub-process is run only once. The next time you run your pipeline, "suggest_feature_types" will be skipped and your main process will proceed to the end.\n\n""")
        exit()


####################################
#         More Toy Data
####################################
letters = {
    'lower_case':string.ascii_lowercase,
    'upper_case':string.ascii_uppercase,
    'digits':string.digits,
    'punctuation': string.punctuation,
}
def get_random_string(length):
    choices = letters['upper_case'] + letters['digits'] + letters['punctuation']
    rstring = ''.join(random.choice(choices) for i in range(length))
    return rstring

class MixedTypeDF():
    ####### Mixed-type Data Frame #######
    # Dataframes with columns containing two types:
    # 1. numbers considered as float and
    # 2. anything else to be considered as ordinals
    def __init__(self, nrows=64, 
        TARGET_CLASSES=[0,1,2], 
        TARGET_CLASSES_PROBABILITIES= [0.6,0.3,0.1], 
        columns_setting=None,
        target_based_transform=None,):
        super(MixedTypeDF, self).__init__()

        self.TARGET_CLASSES = TARGET_CLASSES
        self.TARGET_CLASSES_PROBABILITIES =TARGET_CLASSES_PROBABILITIES
        
        if nrows is None: nrows = 64
        self.nrow = nrows

        if columns_setting is None:
            """
            dtype: only str or float (let bool, ordinals etc be str)
            """
            columns_setting = {
                "name": {'dtype': 'str', 'dist': 'unique', 'prefix': 'name'},
                "gender": {'dtype':'str', 'dist': 'discrete', 
                    'choice': ['M','F','O'], 'p':[0.45,0.45,0.1],
                    'n_missing': 3, 'n_invalid': 3,
                },
                "score1": {'dtype': 'float', 'dist' : 'uniform continuous',
                    'params': {'min': 70, 'max':80}, 
                    'n_missing': 3,
                },
                "smoking": { 'dtype': 'str', 'dist': 'discrete',
                    'choice': ['yes', 'no'], 'p':[0.1,0.9], 
                    'n_missing': 10, 'n_invalid': 2,
                },
                "score2": {'dtype': 'float', 'dist' : 'normal',
                    'params': {'mean': 4, 'sd':2.0},
                },
                'target': { 'dtype': 'str', 'dist': 'discrete',
                    'choice': self.TARGET_CLASSES, 
                    'p': self.TARGET_CLASSES_PROBABILITIES, 
                }
            }
        self.columns_setting = columns_setting

        if target_based_transform is None:
            self.target_based_transform = self.default_transform

    def default_transform(self, df):
        df['score2'] = df['score2'] + (df['target']==1)*1.0
        df['score2'] = df['score2'] + (df['target']==2)*4.0
        df['score1'] = df['score1'] - (df['target']==1)*5.
        df['score1'] = df['score1'] - (df['target']==2)*10.
        df.loc[df['target']==2, 'smoking'] = "yes"
        return df

    def get_mixed_type_df_random_dataframe(self):
        df = {}
        for col, setting in self.columns_setting.items():
            df[col] = self.get_one_column(setting)
        df = pd.DataFrame(df)  

        df = self.target_based_transform(df)  

        for col, setting in self.columns_setting.items():
            self.corrupt_one_column(df, col, setting)
        return df
        
    def get_one_column(self, setting):
        s = setting
        if s['dist'] == 'unique':
            arr = [s['prefix'] + f'{i+1}' for i in range(self.nrow)] 
        elif s['dist'] == 'discrete':
            arr = np.random.choice(s['choice'], p=s['p'], size=(self.nrow,))
        elif s['dist'] == 'uniform continuous':
            pr = s['params']
            arr = np.random.uniform(pr['min'], pr['max'], size=(self.nrow,))
        elif s['dist'] == 'normal':
            pr = s['params']
            arr = np.random.normal(pr['mean'], pr['sd'], size=(self.nrow,))
        return arr

    def corrupt_one_column(self, df, col, setting):
        s = setting
        noutliers = 0
        if 'n_missing' in s: 
            noutliers += s['n_missing']
        else: 
            s['n_missing'] = 0
        if 'n_invalid' in s: 
            noutliers += s['n_invalid']
        else: 
            s['n_invalid'] = 0

        if noutliers > 0 :
            outliers = np.random.choice(range(self.nrow), size=(noutliers,), replace=False)

            if s['dtype'] == 'float':
                df.loc[outliers,col] = [None for _ in range(len(outliers))]
            if s['dtype'] == 'str':
                missing_indices = outliers[:s['n_missing']]
                invalid_indices = outliers[s['n_missing']:]

                df.loc[missing_indices,col] = ['' for _ in range(len(missing_indices))]
                df.loc[invalid_indices,col] = [get_random_string(7) for _ in range(len(invalid_indices))]           

####################################
#            Utils
####################################

def get_dominant_type(arr):
    tmp_ = list(arr)
    typecounter = {type(x):0 for x in set(tmp_)}
    for x in tmp_: typecounter[type(x)]+=1

    dominant, ndom = None, -1
    for t, n in typecounter.items():
        if n > ndom: 
            dominant = t
            ndom = n
    return dominant, typecounter

def dataframe_dominant_types(df, features):
    domtypes, counters = {}, {}
    for col in features:
        domtypes[col], counters[col] = get_dominant_type(df[col])
    """ domtypes is like
    {'name': <class 'str'>, 
        'gender': <class 'str'>, 
        'score1': <class 'float'>, 
        'smoking': <class 'str'>, 
        'score2': <class 'float'>, 
        'target': <class 'int'>}    
    counters is like
    {'name': {<class 'str'>: 32}, 
        'gender': {<class 'str'>: 32}, 
        'score1': {<class 'NoneType'>: 3, <class 'float'>: 29}, 
        'smoking': {<class 'str'>: 32}, 
        'score2': {<class 'float'>: 32}, 
        'target': {<class 'int'>: 32}}    
    """
    return domtypes, counters

def dataframe_suggested_types(df, TARGET_LABEL_NAME=None, float_threshold=1024, token_threshold=16):
    """
    df: dataframe
    if TARGET_LABEL_NAME is not None, exclude the target column 
      (assuming its column name is TARGET_LABEL_NAME)

    suggested_types is like:
    {
        NUMERICAL_FEATURES : ['score1', 'score2']
        TOKEN_FEATURES : ['name', 'gender', 'smoking', 'target']
        _TBD_ : []
        TARGET_LABEL_NAME : 'target'    
    }
    """

    string_type = type("hello")
    float_type = type(0.123)
    int_type = type(1)

    features = [f for f in df.columns]
    suggested_types = {'NUMERICAL_FEATURES':[], 'TOKEN_FEATURES':[], '_TBD_':[]}
    if TARGET_LABEL_NAME is not None:
        try:
            features.remove(TARGET_LABEL_NAME) 
        except:
            print('\n :: TARGET_LABEL_NAME invalid? :: ')
            print(f'Check your TARGET_LABEL_NAME={TARGET_LABEL_NAME}. It should refer to the column of a dataframe (like csv file) that contains the ground-truth classification target/label. Exiting the process now.\n')
            exit()

        suggested_types['TARGET_LABEL_NAME'] = f"'{TARGET_LABEL_NAME}'"

    domtypes, _ = dataframe_dominant_types(df, features)
    for feature, classtype in domtypes.items():
        if classtype == string_type:
            ftype = "token"
        elif classtype in [float_type, int_type]:
            variation = len(set(df[feature]))
            if variation >= float_threshold:
                ftype = 'numeric'
            elif variation < token_threshold:
                ftype = 'token'
            else:
                ftype = '_TBD_' # to be determined
        else:
            raise NotImplementedError("Data type not recognized. Please raise the issue to developers. We will consider adding the unknown datatype to our pipeline")
        
        if ftype == "token":
            suggested_types["TOKEN_FEATURES"].append(feature)
        elif ftype == "numeric":
            suggested_types["NUMERICAL_FEATURES"].append(feature)
        else:
            suggested_types[ftype].append(feature)           
    return suggested_types