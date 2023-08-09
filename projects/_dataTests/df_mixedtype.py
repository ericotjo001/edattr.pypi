""" 
Dataframes with columns containing two types:
1.  numbers considered as float and
2.  anything else considered as ordinals

No long sentences in any of the dataframe columns. 
For long sentences, we recommend using a separate embedding, or preferably a pretrained one.
"""
import os, argparse
import pandas as pd
import numpy as np
from edattr.data import MixedTypeDF
from edattr.utils import sort_dictionary_by_values_desc 

token_features = ['gender', 'smoking']
numerical_features = ['score1', 'score2']

pd.set_option("display.precision", 2)

def main(**kwargs):
    mt = MixedTypeDF(nrows=kwargs['nrows'], columns_setting=None)
    df = mt.get_mixed_type_df_random_dataframe()
    print_(df)

    print_overview(df)
    """
    gender     {'F': 26, 'O': 16, 'M': 16, '': 3, '5GW@L7)': 1, 'H;(<C+.': 1, '[]*F`KN': 1}
    smoking    {'no': 29, 'yes': 23, '': 10, 'LDSK_4)': 1, "'F<$Q6`": 1}
    score1     {'mean': 68.96, 'sd': 17.36}
    score2     {'mean': 3.73, 'sd': 1.78}
    target     {1, 2, 3}    
    """
    
    from edattr.data import collect_vocabulary
    word_to_ix = collect_vocabulary(df, token_features)
    print('\nVocabularies:')
    print(word_to_ix)

def print_(df):
    if len(df)<=64:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(df)
    else:
        print(df)

def print_overview(df):
    TARGET_LABEL_NAME = 'target'
    for tokf in token_features:
        counters = {}
        for x in set(list(df[tokf])):
            counters[x] = len(df[df[tokf]==x])
        print('%-10s'%(tokf), sort_dictionary_by_values_desc(counters))
    for numf in numerical_features:
        simplestats = {
            'mean': np.round(np.mean(df[numf]),2), 
            'sd': np.round(np.var(df[numf])**0.5,2),
        }
        print('%-10s'%(numf), simplestats)
    print('%-10s'%('target'), set(list(df['target'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)    
    
    parser.add_argument('--nrows', default=None, type=int, help=None)

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary        

    main(**kwargs)