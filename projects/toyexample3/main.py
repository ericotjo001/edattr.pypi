import os, argparse
from setup import *
from toydata3 import prep_toyexample_dir, save_toyexample_csv


#######################################
#      Basic Data Preprocessing 
#######################################
# For classifier task

def prepare_data(**kwargs):
    print('=== prepare toy data ===')
    v = kwargs['verbose']

    # Let's create a raw toy data, if not yet created
    # By default, we put our data somewhere in "Desktop/edattr.ws/data"
    DATA_DIR = prep_toyexample_dir(**kwargs)
    if not os.path.exists(DATA_DIR): save_toyexample_csv(DATA_DIR)    

    # We like to create the directory variable explicitly. Why not?
    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    # Load our raw toy data - see above (assume it's in the form of csv file)
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

# #######################################
# #           Main Processes
# #######################################
def toy3_standard_train_val_test_endorse(parser):
    parser.add_argument('--toggle', default=None, type=str, help=None)  
    parser.add_argument('--n_epochs', default=4, type=int, help=None)  
    parser.add_argument('--batch_size', default=32, type=int, help=None)

    parser.add_argument('--DEV_ITER', default=0, type=int, help=None)

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary    

    kwargs.update({
        'prefix': 'branch',
        'feature_mode':'Token+Num',
        'DIRECTORY_MODE': 'singlefile',
        })

    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    # imported from setup.py
    cf = classifierToy3(DIRS, **kwargs)

    from edattr.endorse import StandardClassifierEndorsementVis
    cfev = StandardClassifierEndorsementVis(DIRS,**kwargs)    

    TOGGLE = kwargs['toggle']
    if TOGGLE == '0':
        cf.log_model_number_of_params(**kwargs); exit()
    
    if TOGGLE is None: TOGGLE = '11111'

    if TOGGLE[0] == '1':
        cf.train_val_test(verbose=kwargs['verbose'])
        cf.visualize_output()
    if TOGGLE[1] == '1':
        cf.endorse_selected_models(**kwargs)
        cfev.visualize_endorsement_selected_models(**kwargs)  
    if TOGGLE[2] == '1':        
        cf.eec_partition_selected_models(**kwargs)   
        cf.eec_selected_models(**kwargs)   
    if TOGGLE[3] == '1':
        cf.post_eec_train_val_test(**kwargs)
        cf.visualize_post_eec_output(**kwargs)

    # Additional section: if you want to compare with common ML algorithms 
    if TOGGLE[4] == '1':
        from edattr.cmlpackage import StandardS2CommonMLClassifierPipeline
        cml = StandardS2CommonMLClassifierPipeline(cf,TARGET_LABEL_DICTIONARY=None)
        cml.train_test_common_ml_classifiers()

        cf.kwargs['compare-common-ml'] = True
        cf.visualize_post_eec_output(**kwargs)

def toy3_standard_aggregate(parser):
    print('toy3_standard_aggregate...')
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary    
    kwargs.update({'prefix':'branch','label': 'dummylabel'}) 

    label_suffix = 'toy3_standard'
    from edattr.aggregator import aggregate_val_test_results
    aggregate_val_test_results(label_suffix=label_suffix, **kwargs)

    from edattr.setup_template import DataSetupTypeS2, DatasetTypeS2
    from edattr.aggregator import aggregate_endorsement_samples, aggregate_endorsement_samples_vis
    kwargs.update({
        'TARGET_LABEL_NAME': TARGET_LABEL_NAME,
        'TOKEN_FEATURES':TOKEN_FEATURES,
        'NUMERICAL_FEATURES':NUMERICAL_FEATURES,
        'feature_mode': 'Token+Num',
        'endorsement_mode' : 'shap-lime-top2',
        })
    aggregate_endorsement_samples(DataSetupTypeS2, DatasetTypeS2, 
        label_suffix=label_suffix, **kwargs)
    aggregate_endorsement_samples_vis(label_suffix=label_suffix, **kwargs)

    from edattr.factory import clean_up_directory
    clean_up_directory(**kwargs)  # let's do a bit of cleanup, delete folder with 'dummylabel'


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

    parser.add_argument('--DIRECTORY_MODE', default='singlefile', type=str, help=None)
    parser.add_argument('--DATA_FILE_NAME', default='toyexample3.csv', type=str, help=None)

    parser.add_argument('--DATA_DIR', default=None, type=str, help=DATA_DIR_INFO)
    parser.add_argument('--DATA_PROCESSING_DIR', default=None, type=str, help=DATA_PROCESSING_DIR_INFO)

    parser.add_argument('--full_projectname', default='toyclassifier3', type=str, help=LABEL_INFO)
    parser.add_argument('--label', default=None, type=str, help=None)   

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    if kwargs['mode'] == 'preprocess_normalize':
        prepare_data(**kwargs)
    elif kwargs['mode'] == 'standard':
        toy3_standard_train_val_test_endorse(parser)
    elif kwargs['mode'] == 'standard_aggregate':
        toy3_standard_aggregate(parser)
    else:
        raise NotImplementedError('invalid mode?')
