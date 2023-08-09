import os, argparse
from setup import *

# Note: you may want to refer to our projects/toyexample2 first

#######################################
#      Basic Data Preprocessing 
#######################################
# For classifier task

def prepare_data(**kwargs):
    print('=== prepare maternal health data ===')
    v = kwargs['verbose']

    # We like to create the directory variable explicitly. Why not?
    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    # Load our medical maternal health data
    import pandas as pd
    df = pd.read_csv(DIRS['DATA_DIR'], index_col=False)

    # Do some pre-processing of your csv data here
    from edattr.data import DataFramePreProcessorTypeA
    dfpp = DataFramePreProcessorTypeA() # create the processor
    dfpp.suggest_feature_types(df, DIRS['DATA_CACHE_DIR']+'-feature.txt', 
        TARGET_LABEL_NAME=TARGET_LABEL_NAME)
    
    processed_ = dfpp.process_dataframe(df, DIRS['DATA_CACHE_DIR'], 
        TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME, verbose=v)
    print('Data cache at', DIRS['DATA_CACHE_DIR'])

    # Do visualization here
    from edattr.data import DataVis
    datavis = DataVis(DIRS['DATA_VIS_DIR']) # create the visualization object
    df_processed = processed_['df_processed']

    relevant_columns = list(df_processed.columns) + [TARGET_LABEL_NAME]
    datavis.vis_mixed_types(df[relevant_columns], df_processed, TARGET_LABEL_NAME=TARGET_LABEL_NAME, TOKEN_FEATURES=TOKEN_FEATURES, word_to_ix=processed_['word_to_ix'], verbose=v)
    print('Data visualization at', DIRS['DATA_VIS_DIR'])


#######################################
#           Main Process
#######################################

def maternalhealth_kfold_train_val_test_endorse(parser):
    parser.add_argument('--toggle', default=None, type=str, help=None)  
    parser.add_argument('--n_epochs', default=128, type=int, help=None)  
    parser.add_argument('--batch_size', default=32, type=int, help=None)

    parser.add_argument('--DEV_ITER', default=0, type=int, help=None)

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary
    kwargs.update({'prefix':'k'})    

    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    # imported from setup.py
    kf = kFold_matHealth(DIRS, **kwargs)

    from edattr.endorse import kFoldClassifierEndorsementVis
    kfev = kFoldClassifierEndorsementVis(DIRS,**kwargs)    

    TOGGLE = kwargs['toggle']
    if TOGGLE == '0':
        kf.log_model_number_of_params(**kwargs); exit()    
    if TOGGLE is None: TOGGLE = '11111'

    kwargs.update({
        'feature_mode':'Token+Num',
        })

    if TOGGLE[0] == '1':
        kf.train_val_test(verbose=kwargs['verbose'])
        kf.visualize_output()
    if TOGGLE[1] == '1':
        kf.endorse_selected_models(**kwargs)
        kfev.visualize_endorsement_selected_models(**kwargs)  
    if TOGGLE[2] == '1':        
        kf.eec_partition_selected_models(**kwargs)   
        kf.eec_selected_models(**kwargs)   
    if TOGGLE[3] == '1':
        kf.post_eec_train_val_test(**kwargs)
        kf.visualize_post_eec_output(**kwargs)

    # Additional section: if you want to compare with common ML algorithms 
    if TOGGLE[4] == '1':
        from edattr.cmlpackage import kFoldK2CommonMLClassifierPipeline
        cml = kFoldK2CommonMLClassifierPipeline(kf,TARGET_LABEL_DICTIONARY=TARGET_LABEL_DICTIONARY)
        cml.train_test_common_ml_classifiers()

        kf.kwargs['compare-common-ml'] = True
        kf.visualize_post_eec_output(**kwargs)

#######################################
#           Others
#######################################

def maternalhealth_kfold_aggregate(parser):
    print('maternalhealth_aggregate...')
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary  
    kwargs.update({'prefix':'k', 'label':'dummylabel'})   

    label_suffix = 'maternalhealth_kfold'

    from edattr.aggregator import aggregate_val_test_results
    aggregate_val_test_results(label_suffix=label_suffix, **kwargs)

    from edattr.setup_template import DataSetupTypeK2
    from edattr.aggregator import aggregate_endorsement_samples, aggregate_endorsement_samples_vis
    kwargs.update({
        'TARGET_LABEL_NAME': TARGET_LABEL_NAME,
        'TOKEN_FEATURES':TOKEN_FEATURES,
        'NUMERICAL_FEATURES':NUMERICAL_FEATURES,
        'feature_mode': 'Token+Num',
        'endorsement_mode' : 'shap-lime-top2',
        })
    aggregate_endorsement_samples(DataSetupTypeK2, matHealth_kfold_dataset, 
        label_suffix=label_suffix, **kwargs)
    aggregate_endorsement_samples_vis(label_suffix=label_suffix, **kwargs)

    from edattr.factory import clean_up_directory
    clean_up_directory(**kwargs)  # let's do a bit of cleanup, delete folder with 'dummylabel'

def maternalhealth_compare(parser):
    parser.add_argument('--best.metric', default='acc', type=str, help="Choose the model with this best metric for comparison.")
    parser.add_argument('--split', default='train', type=str, help='train/val/test')

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary  
    kwargs.update({'prefix':'k'})  
    
    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    # imported from setup.py
    kf = kFold_matHealth(DIRS, **kwargs)
    kf.compare_edattr_batchwise()
    kf.visualize_edattr_comparison()

    kf.compare_edattr_batchwise(absval=True)
    kf.visualize_edattr_comparison(absval=True)
    
def maternalhealth_compare_aggregate(parser):
    print('maternalhealth_compare_aggregate...')
    parser.add_argument('best.metric', default='acc', type=str, help=None)
    parser.add_argument('--label_suffix', default='maternalhealth_kfold', type=str, help=None)

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary  

    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    from edattr.endorse_plugin import aggregate_compare_results
    aggregate_compare_results(DIRS, **kwargs)
    aggregate_compare_results(DIRS, absval=True, **kwargs)

WORKSPACE_DIR_INFO = """Default workspace is at ~/Desktop/edattr.ws"""
DATA_DIR_INFO = """Full directory, if specified. Otherwise, consider putting the csv file into <WORKSPACE_DIR>/data"""
DATA_PROCESSING_DIR_INFO = "Where to store the results of preprocessing your data."
LABEL_INFO = "Label determines the preset configuration (see setup.py)."

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('-m','--mode', default=None, type=str, help=None)
    parser.add_argument('-v','--verbose', default=100, type=int, help=None)
    parser.add_argument('-w','--WORKSPACE_DIR', default=None, type=str, help=WORKSPACE_DIR_INFO)

    parser.add_argument('--DATA_FILE_NAME', default='Maternal Health Risk Data Set.csv', type=str, help=None)
    parser.add_argument('--DATA_DIR', default=None, type=str, help=DATA_DIR_INFO)
    parser.add_argument('--DATA_PROCESSING_DIR', default=None, type=str, help=DATA_PROCESSING_DIR_INFO)

    parser.add_argument('--full_projectname', default='maternalhealth', type=str, help=LABEL_INFO)
    parser.add_argument('--label', default=None, type=str, help=None)   

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    if kwargs['mode'] == 'preprocess_normalize':
        prepare_data(**kwargs)
    elif kwargs['mode'] == 'kfold':
        maternalhealth_kfold_train_val_test_endorse(parser)
    elif kwargs['mode'] == 'kfold_aggregate':
        maternalhealth_kfold_aggregate(parser)
    elif kwargs['mode'] == 'compare':
        maternalhealth_compare(parser)
    elif kwargs['mode'] == 'aggregate_compare':
        maternalhealth_compare_aggregate(parser)        
    else:
        raise NotImplementedError('invalid mode?')
