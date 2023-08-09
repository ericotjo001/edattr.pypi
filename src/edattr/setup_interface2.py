from .utils import *
from .decorator import *

from .setup_template import DatasetTypeK2, DataSetupTypeK2, DataSetupTypeS2, DatasetTypeS2
from .setup_interface1 import kFoldClassifier, DatasetSingleClassifierCSV, StandardClassifier, DatasetStandardClassifierCSV

from .endorse_plugin import ComparisonEndorsementClassifierK2, ComparisonEndorsementClassifierS2

######### Standard #########
# Set up data from CSV for standard train/val/test

""" *** S2 series ***
Data Frame Types: TokenAndFloat DATAFRAME (see data.py)
"""
class DatasetStandardClassifierCSVTypeS2(DatasetStandardClassifierCSV):
    def __init__(self, setupTypeS2, split):    
        super(DatasetStandardClassifierCSVTypeS2, self).__init__(setupTypeS2, split)

    def __getitem__(self, i):
        """
        "indices" is a variable introduced by our kfold setup. 
        If there are n total rows in the CSV file,  then self.indices will be a subset of 
          [0,1,...,n-1] that depends on your split (train/val/test)
        """
        idx = self.indices[i] # raw index from the CSV file
        
        tokens = self.df[self.TOKEN_FEATURES].loc[idx].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[idx].to_numpy()
        x = np.concatenate((tokens, numerics))

        y0 = self.df_target[idx]
        return idx, x, y0  
        
class StandardClassifierS2(StandardClassifier, ComparisonEndorsementClassifierS2):
    def __init__(self, DIRS, **kwargs):
        super(StandardClassifierS2, self).__init__(DIRS, **kwargs)

    def set_dataset_object(self):
        self.dataset_object = DatasetStandardClassifierCSVTypeS2
        self.eec_dataset_object = DatasetSingleClassifierCSV

    def set_dataset(self):
        self.dataset = DataSetupTypeS2(
            self.DIRS, 
            TARGET_LABEL_NAME=self.TARGET_LABEL_NAME, 
            TOKEN_FEATURES=self.TOKEN_FEATURES,
            NUMERICAL_FEATURES=self.NUMERICAL_FEATURES,
            **self.kwargs)                    


#########  kfold   #########
# Set up data from CSV for kfold train/val/test


""" *** K2 series ***
Data Frame Types: TokenAndFloat DATAFRAME (see data.py)
"""
class DatasetKFoldClassifierTypeK2CSV(DatasetTypeK2):
    def __init__(self, setupTypeK2, k, split):         
        # Setting df and df_target up using the parent class (data loaded by setupTypeK2)
        super(DatasetKFoldClassifierTypeK2CSV, self).__init__(setupTypeK2, k, split)

    def __getitem__(self, i):
        """
        "indices" is a variable introduced by our kfold setup. 
        If there are n total rows in the CSV file,  then self.indices will be a subset of 
          [0,1,...,n-1] that depends on your split (train/val/test)
        """
        idx = self.indices[i] # raw index from the CSV file
        
        tokens = self.df[self.TOKEN_FEATURES].loc[idx].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[idx].to_numpy()
        x = np.concatenate((tokens, numerics))

        y0 = self.df_target[idx]
        return idx, x, y0             

class kFoldClassifierK2(kFoldClassifier, ComparisonEndorsementClassifierK2):
    def __init__(self, DIRS, **kwargs):
        super(kFoldClassifierK2, self).__init__(DIRS, **kwargs)

    def set_dataset_object(self):
        self.dataset_object = DatasetKFoldClassifierTypeK2CSV
        self.eec_dataset_object = DatasetSingleClassifierCSV

    def set_dataset(self):  
        self.number_of_folds = self.config['kfold']
        self.dataset = DataSetupTypeK2(
            self.DIRS, 
            TARGET_LABEL_NAME=self.TARGET_LABEL_NAME, 
            TOKEN_FEATURES=self.TOKEN_FEATURES,
            NUMERICAL_FEATURES=self.NUMERICAL_FEATURES,
            kfold=self.number_of_folds,
            **self.kwargs)  


