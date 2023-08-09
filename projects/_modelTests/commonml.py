import time
import numpy as np
import pandas as pd

n = 360
def get_df(n):
    df = {
        'x1': np.random.normal(0,1,size=(n,)),
        'x2': np.random.normal(10,1,size=(n,)),
        'x3': np.random.normal(5,4,size=(n,)),
        't1': np.random.choice([1,2], size=(n,)),
        't2': np.random.choice([3,4,5], size=(n,)),
        'target': np.zeros(shape=(n,)),
    }
    df['target'] = (df['t1']==1).astype(int) + (df['t2']==3).astype(int) + (df['x3']>6).astype(int) + (df['x2']<1.4).astype(int)
    df = pd.DataFrame(df)
    return df

df = get_df(n)
features = [f for f in df.columns if not f=='target']
print(features)

#######
n_test = 1000
df_test = get_df(n_test)

def train_and_test(X, Y, X_test, Y_test, model, model_config, label):
    print(f'======= {label} =======')
    clf = model(**model_config)
    
    start = time.time()
    clf.fit(X,Y)
    end = time.time()
    t = end-start

    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred==Y_test)
    print(f'  {label} -> acc:{round(acc,2)} [[time: {round(t,2)}s = {round(t/60.,2)} min]]', )

from sklearn.linear_model import LogisticRegression
train_and_test(df[features], df['target'], 
    df_test[features], df_test['target'],
    LogisticRegression, {}, "Logistic L2")

from sklearn.neighbors import KNeighborsClassifier
train_and_test(df[features], df['target'], 
    df_test[features], df_test['target'],
    KNeighborsClassifier, {'n_neighbors':5}, "KNeighborsClassifier")

from sklearn.ensemble import RandomForestClassifier
train_and_test(df[features], df['target'], 
    df_test[features], df_test['target'],
    RandomForestClassifier, {'max_depth':7},"RandomForestClassifier")

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
rbfs = RBFSampler(gamma=1)
X_features = rbfs.fit_transform(df[features])
X_test_features = rbfs.fit_transform(df_test[features])
train_and_test(X_features, df['target'], 
    X_test_features, df_test['target'],
    SGDClassifier, {'max_iter':1000, 'tol':1e-3},"RBFSampler")

####### unused #######

# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# train_and_test(df[features], df['target'], 
#     df_test[features], df_test['target'],
#     GaussianProcessClassifier, {'kernel':1.0 * RBF(1.0)},"RBF") 
# need too much memory, use kernel_approx version

# from sklearn import svm
# train_and_test(df[features], df['target'], 
#     df_test[features], df_test['target'],
#     svm.SVC, {}, "SVM") # too slow for large samples, like 360k

