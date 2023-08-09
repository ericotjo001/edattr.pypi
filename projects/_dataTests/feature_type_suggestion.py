import argparse
import numpy as np
import pandas as pd

"""
feature_type_suggestion
      name   gender     score1  smoking    score2  target
0    name1        M  82.244724      yes  6.753786       0
1    name2           84.986232      yes -0.236413       1
2    name3        M  49.367405  M*0K@$)  5.218483       1
3    name4        F       None       no  2.111683       0
...
30  name31        M  55.356343      yes -0.884893       1
31  name32        M  92.431033       no  9.291112       0

Suggested types:
  NUMERICAL_FEATURES = ['score1', 'score2']
  TOKEN_FEATURES     = ['name', 'gender', 'smoking', 'target']
  _TBD_              = []
  TARGET_LABEL_NAME  = 'target'
"""

N_DATA = 32
TARGET_LABEL_NAME = 'target'

from edattr.data import MixedTypeDF, dataframe_suggested_types
# let's create some synthetic data
mt = MixedTypeDF(nrows=N_DATA, 
    TARGET_CLASSES=[0,1,2], 
    TARGET_CLASSES_PROBABILITIES= [0.6,0.3,0.1], 
    columns_setting=None)


def main(**kwargs):
    print('feature_type_suggestion')
    df = mt.get_mixed_type_df_random_dataframe() 
    print(df)

    print('\nSuggested types:')
    suggested_types = dataframe_suggested_types(df, TARGET_LABEL_NAME=TARGET_LABEL_NAME)
    for feature_name, suggested_type in suggested_types.items():
        print(f'{"  %-18s"%(str(feature_name))} = {suggested_type}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)    
    
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary        

    main(**kwargs)