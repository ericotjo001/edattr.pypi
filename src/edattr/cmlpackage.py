# CML: common machine learning packages
from .utils import *

from .metric import compute_classification_metrics

class CommonMLClassifierPipeline():
    def __init__(self, clf, TARGET_LABEL_DICTIONARY=None):
        super(CommonMLClassifierPipeline, self).__init__()
        self.clf = clf
        self.TARGET_LABEL_DICTIONARY = TARGET_LABEL_DICTIONARY

    @staticmethod
    def train_and_test(X, Y, X_test, Y_test, model, model_config, label):
        clf = model(**model_config)        
        clf.fit(X,Y)

        y_pred = clf.predict(X_test)     

        confusion_matrix = compute_classification_metrics(y_pred,Y_test)
        return confusion_matrix

    def prep_data_common_ml_classifiers(self, branch):
        raise NotImplementedError('Implement Downstream')

    def train_test_common_ml_classifiers(self):
        clf = self.clf
        prefix = clf.kwargs['prefix']

        common_ml_results = {}
        common_ml_eec_results = {}
        best_values_where = clf.load_best_value_where()
        for metric_type, record in best_values_where.items():
            branch = record[0] # k-th fold
            X, Y, X_test, Y_test = self.prep_data_common_ml_classifiers(branch)
            
            label_ = f"{prefix}-{branch}-best.{metric_type}"
            common_ml_eec_results[label_] = {}
            self.train_eec_test_common_ml_classifiers(common_ml_eec_results, X_test, Y_test, prefix, branch, metric_type, label_)

            common_ml_results[label_] = self.train_test_common_ml_classifiers_(X, Y, X_test, Y_test) 

        CML_RESULT_DIR = os.path.join(clf.DIRS['TEST_RESULT_DIR'], 'common_ml_results.json')
        with open(CML_RESULT_DIR, 'w') as json_file:
            json.dump(common_ml_results, json_file, indent=4, sort_keys=True)
        print(f"--> saved to {CML_RESULT_DIR}")

        CML_EEC_ESULT_DIR = os.path.join(clf.DIRS['EEC_RESULT_DIR'], 'common_ml_eec_results.json')
        with open(CML_EEC_ESULT_DIR, 'w') as json_file:
            json.dump(common_ml_eec_results, json_file, indent=4, sort_keys=True)
        print(f"--> saved to {CML_EEC_ESULT_DIR}")

    def train_test_common_ml_classifiers_(self, X, Y, X_test, Y_test):
        from warnings import simplefilter
        from sklearn.exceptions import ConvergenceWarning
        simplefilter("ignore", category=ConvergenceWarning)

        confusion_matrices_by_model = {}

        from sklearn.linear_model import LogisticRegression
        label  = "LogisticRegression"
        confusion_matrices_by_model[label] = self.train_and_test(X, Y, X_test, Y_test, 
            LogisticRegression, {}, label)

        from sklearn.neighbors import KNeighborsClassifier
        label = 'KNeighborsClassifier' 
        confusion_matrices_by_model[label] = self.train_and_test(X, Y, X_test, Y_test, 
            KNeighborsClassifier, {'n_neighbors':5}, label)

        from sklearn.ensemble import RandomForestClassifier
        label = "RandomForestClassifier"
        confusion_matrices_by_model[label] = self.train_and_test(X, Y, X_test, Y_test,
            RandomForestClassifier, {'max_depth':7}, label)

        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import SGDClassifier
        label = 'RBFSampler'
        rbfs = RBFSampler(gamma=1)
        X_features = rbfs.fit_transform(X)
        X_test_features = rbfs.fit_transform(X_test)
        confusion_matrices_by_model[label] = self.train_and_test(X_features, Y,  X_test_features, Y_test, SGDClassifier, {'max_iter':1000, 'tol':1e-3},label) 

        return confusion_matrices_by_model   

    def train_eec_test_common_ml_classifiers(self, common_ml_eec_results, 
        X_test, Y_test, prefix, branch, metric_type, label_):
        clf = self.clf
        for eec_subtype in clf.config['eec_modes']:
            EEC_FOLDER_LABEL_NAME = f'{prefix}-{branch}.best.{metric_type}.partition.{eec_subtype}'
            EEC_FOLDER_LABEL_DIR = os.path.join(clf.DIRS['EEC_RESULT_DIR'], EEC_FOLDER_LABEL_NAME)
            EEC_TRAINSET_DIRS = os.path.join(EEC_FOLDER_LABEL_DIR,'eec-train-data-*.csv')

            common_ml_eec_results[label_][eec_subtype] = {}
            for eec_train_dir in glob.glob(EEC_TRAINSET_DIRS):
                eec_train_base_filename = os.path.basename(eec_train_dir)
                # eec_train_dir is like "path/to/eec-train-data-t8.csv"
                # so eec_train_base_filename is like eec-train-data-t8.csv

                tlabel = eec_train_base_filename.replace('-','.').split('.')[-2]

                common_ml_eec_results[label_][eec_subtype][tlabel] = self.train_eec_test_common_ml_classifiers_(eec_train_dir, X_test, Y_test)

    # def train_eec_test_common_ml_classifiers_(self, eec_train_dir, X_test, Y_test):
    #     raise NotImplementedError('Implement Downstream')

    def train_eec_test_common_ml_classifiers_(self, eec_train_dir, X_test, Y_test):
        df = pd.read_csv(eec_train_dir)
        target_label = 'y0' # y0 is used as a convention
        # in EEC train set, feature names and target label names are alrd standardized to something like f1,f2,...,y0

        features = [f for f in df.columns if f!=target_label]
        X = df.loc[:, features].to_numpy()
        Y = df.loc[:, target_label].to_numpy()

        cm_by_model = {-1:-1} # a placeholder for None
        if set(Y) == set(Y_test):
            cm_by_model = self.train_test_common_ml_classifiers_(X, Y, X_test, Y_test)
        # So, if the EEC are missing some labels, we don't really want them
        # That's why we have the above placeholder for None

        return cm_by_model


""" ======= kFoldCommonMLClassifierPipeline =======
kfold classification, more suitable for smaller dataset.
Data Frame Types: NAIVE NUMERICAL DATAFRAME (see data.py)
"""

class kFoldCommonMLClassifierPipeline(CommonMLClassifierPipeline):
    def __init__(self, clf, TARGET_LABEL_DICTIONARY=None):
        super(kFoldCommonMLClassifierPipeline, self).__init__(clf, TARGET_LABEL_DICTIONARY=TARGET_LABEL_DICTIONARY)

    def prep_data_common_ml_classifiers(self, branch):
        clf = self.clf

        train_dataset_ = clf.get_dataset_(clf.dataset_object, branch, 'train')
        # print(train_dataset_.df.shape) # full df, already normalized
        # print(train_dataset_.indices, len(train_dataset_.indices)) # indices of only train set

        X = train_dataset_.df[train_dataset_.indices,:]
        Y = train_dataset_.df_target[train_dataset_.indices]
        assert(X.shape[0] == len(Y))

        test_dataset_ =  clf.get_dataset_(clf.dataset_object, branch, 'test')
        X_test = test_dataset_.df[test_dataset_.indices,:]
        Y_test = test_dataset_.df_target[test_dataset_.indices] # raw labels here!

        if self.TARGET_LABEL_DICTIONARY is not None:
            def transform_label(y):
                return self.TARGET_LABEL_DICTIONARY[y]
            Y_test = Y_test.apply(transform_label)
            
        assert(X_test.shape[0] == len(Y_test))    
        return X, Y, X_test, Y_test


""" ======= kFoldK2CommonMLClassifierPipeline =======
kfold classification, more suitable for smaller dataset.
Data Frame Types: TokenAndFloat DATAFRAME (see data.py)
"""

class kFoldK2CommonMLClassifierPipeline(CommonMLClassifierPipeline):
    def __init__(self, clf, TARGET_LABEL_DICTIONARY=None):
        super(kFoldK2CommonMLClassifierPipeline, self).__init__(clf, TARGET_LABEL_DICTIONARY=TARGET_LABEL_DICTIONARY)

    def prep_data_common_ml_classifiers(self, branch):
        clf = self.clf

        train_dataset_ = clf.get_dataset_(clf.dataset_object, branch, 'train')
        # print(train_dataset_.df.shape) # full df, NUMERICAL_FEATURES already normalized. TOKEN_FEATURES already tokenized (int). The rest are left as they are 
        # print(train_dataset_.indices, len(train_dataset_.indices)) # indices of only train set

        X = train_dataset_.df.loc[train_dataset_.indices, list(clf.TOKEN_FEATURES) + list(clf.NUMERICAL_FEATURES)].to_numpy()
        Y = train_dataset_.df_target.loc[train_dataset_.indices]
        assert(X.shape[0] == len(Y))

        test_dataset_ =  clf.get_dataset_(clf.dataset_object, branch, 'test')
        X_test = test_dataset_.df.loc[test_dataset_.indices, list(clf.TOKEN_FEATURES) + list(clf.NUMERICAL_FEATURES)].to_numpy()
        Y_test = test_dataset_.df_target.loc[test_dataset_.indices]

        if self.TARGET_LABEL_DICTIONARY is not None:
            def transform_label(y):
                return self.TARGET_LABEL_DICTIONARY[y]
            Y = Y.apply(transform_label)
            Y_test = Y_test.apply(transform_label)
        assert(X_test.shape[0] == len(Y_test))

        Y = Y.to_numpy() # raw labels here!
        Y_test = Y_test.to_numpy() # raw labels here!

        return X, Y, X_test, Y_test

""" ======= StandardS2CommonMLClassifierPipeline =======
Standard classification, one pass of train/val/test for larger dataset.
Data Frame Types: TokenAndFloat DATAFRAME (see data.py)

"""

class StandardS2CommonMLClassifierPipeline(CommonMLClassifierPipeline):
    def __init__(self, clf, TARGET_LABEL_DICTIONARY=None):
        super(StandardS2CommonMLClassifierPipeline, self).__init__(clf, TARGET_LABEL_DICTIONARY=TARGET_LABEL_DICTIONARY)

    def prep_data_common_ml_classifiers(self, branch):
        clf = self.clf

        train_dataset_ = clf.get_dataset_(clf.dataset_object, 'train')
        # print(train_dataset_.df.shape) # full df, NUMERICAL_FEATURES already normalized. TOKEN_FEATURES already tokenized (int). The rest are left as they are 
        # print(train_dataset_.indices, len(train_dataset_.indices)) # indices of only train set

        X = train_dataset_.df.loc[train_dataset_.indices, list(clf.TOKEN_FEATURES) + list(clf.NUMERICAL_FEATURES)].to_numpy()
        Y = train_dataset_.df_target.loc[train_dataset_.indices]
        assert(X.shape[0] == len(Y))

        test_dataset_ =  clf.get_dataset_(clf.dataset_object, 'test')
        X_test = test_dataset_.df.loc[test_dataset_.indices, list(clf.TOKEN_FEATURES) + list(clf.NUMERICAL_FEATURES)].to_numpy()
        Y_test = test_dataset_.df_target.loc[test_dataset_.indices]

        if self.TARGET_LABEL_DICTIONARY is not None:
            def transform_label(y):
                return self.TARGET_LABEL_DICTIONARY[y]
            Y = Y.apply(transform_label)
            Y_test = Y_test.apply(transform_label)
        assert(X_test.shape[0] == len(Y_test))

        Y = Y.to_numpy() # raw labels here!
        Y_test = Y_test.to_numpy() # raw labels here!

        return X, Y, X_test, Y_test


        