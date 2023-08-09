from .utils import *

def get_param_count_by_label(PROJECT_DIR, label):
    TRAIN_VAL_RESULT_DIR = os.path.join(PROJECT_DIR, label, 'trainval_result')
    nparams = None
    for result_dir in glob.glob(f"{TRAIN_VAL_RESULT_DIR}/trainval-output.*.data"):
        result = joblib.load(result_dir)

        this_n_params = count_parameters(result['model'])
        if nparams is None:
            nparams = this_n_params
        else:
            assert(nparams==this_n_params)
    return nparams

def aggregate_val_test_results(prefix='branch',**kwargs):
    from .factory import manage_dirs
    DIRS = manage_dirs(**kwargs)

    METRICS = ['acc', 'recall', 'precision', 'f1']
    PROJECT_PATHS = glob.glob(os.path.join(DIRS['PROJECT_DIR'],f'{kwargs["label_suffix"]}_*') )
    PROJECT_LABELS = [os.path.basename(x) for x in PROJECT_PATHS]

    agg_df = {'label':[], 'nparams':[]}
    agg_df.update({'branches':[], 'best.mtype':[], }) # branches are like kfold
    agg_df.update({m:[] for m in METRICS})

    for label in PROJECT_LABELS:
        bestvalwhere_dir = os.path.join(DIRS['PROJECT_DIR'], label, 'test_result','bestvalwhere.json')
        try:
            with open(bestvalwhere_dir) as f:
                bestvalwhere = json.load(f)['best_values_where']
        except:
            print(" !! Skipping corrupt or missing file? >>>", 'bestvalwhere.json at', label  )
            continue

        nparams = get_param_count_by_label(DIRS['PROJECT_DIR'], label)        

        for bestm in bestvalwhere:
            agg_df['nparams'].append(nparams)
            
            branch, metric_value = bestvalwhere[bestm] 
            # print(branch, bestm,  metric_value) # like >> 3 acc 0.8020833333333334

            TEST_PATH = os.path.join(DIRS['PROJECT_DIR'], label, 'test_result', f'test-output.{prefix}-{branch}.data')            
            kfold_test_result = joblib.load(TEST_PATH)
            """ like this
            {
            'best.acc': {
                'TP': 26, 'TN': 26, 'FP': 28, 'FN': 16, 'acc': 0.5416666666666666, 'recall': 0.6190476190476191, 'precision': 0.48148148148148145, 'f1': 0.5416666666666666
                }, 
            'best.f1': {
                'TP': 26, 'TN': 26, 'FP': 28, 'FN': 16, 'acc': 0.5416666666666666, 'recall': 0.6190476190476191, 'precision': 0.48148148148148145, 'f1': 0.5416666666666666
                }
            }
            """

            agg_df['label'].append(label)
            agg_df['branches'].append(branch)
            agg_df['best.mtype'].append(f'{bestm}')                    

            item = kfold_test_result['best.'+bestm]
            for m in METRICS:
                agg_df[m].append(round(item[m],3))

    agg_df = pd.DataFrame(agg_df)
    agg_df.to_csv(DIRS['PROJECT_AGGREGATE_DIR'], index=False)
    print(f'aggregate results saved to {DIRS["PROJECT_AGGREGATE_DIR"]}')




##########################################
#
##########################################

def get_features(DATA_CACHE_DIR, feature_mode=None):
    data_cache = joblib.load(DATA_CACHE_DIR)

    if feature_mode is None:
        features = data_cache['features']
    elif feature_mode == 'Token+Num': 
        TOKEN_FEATURES = data_cache['TOKEN_FEATURES']
        NUMERICAL_FEATURES = data_cache['NUMERICAL_FEATURES']
        features = list(TOKEN_FEATURES) + list(NUMERICAL_FEATURES) # yes, in this order     
    return features

def aggregate_endorsement_samples(DataSetup, DataSetObject, **kwargs):
    prefix = kwargs['prefix']
    if prefix == 'k': # kfold
        kwargs.update({'kfold': 5,})
    elif prefix == 'branch': # standard
        pass
    else:
        raise NotImplementedError('unknown prefix')

    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)
    datasetup = DataSetup(DIRS, **kwargs)
    sample_indices = select_random_samples(datasetup,  **kwargs)

    features = get_features(DIRS['DATA_CACHE_DIR'], feature_mode=kwargs['feature_mode'])
   
    if prefix == 'k': # kfold
        dummy_k, dummy_split = 0, 'train'
        dataset = DataSetObject(datasetup, dummy_k, dummy_split)
    elif prefix == 'branch': # standard
        dummy_split = 'train'
        dataset = DataSetObject(datasetup, dummy_split)
    else:
        raise NotImplementedError('unknown prefix')
    dataset.indices = sample_indices

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, shuffle=False, batch_size=len(sample_indices))

    PROJECT_PATHS = glob.glob(os.path.join(DIRS['PROJECT_DIR'],
        f'{kwargs["label_suffix"]}_*') )
    PROJECT_LABELS = [os.path.basename(x) for x in PROJECT_PATHS]    
    for projectdir in PROJECT_PATHS:
        save_endorsement_sample_result(projectdir, dataloader, features, **kwargs)


def select_random_samples(datasetup, **kwargs):
    # print(datasetup.df.shape) # numpy array (n,D)
    # print(datasetup.df_target.shape) # numpy array (n,)
    prefix = kwargs['prefix']

    TARGET_LABEL_NAME = kwargs['TARGET_LABEL_NAME']

    sample_indices = []
    for c in set(datasetup.df_target):
        label = c # it's raw label, as found in the csv file!

        idx = np.where(datasetup.df_target==label)[0].astype(int)
        sample_indices = sample_indices + list(idx[:4])
    return sample_indices

def save_endorsement_sample_result(projectdir, dataloader, features, **kwargs):
    prefix = kwargs['prefix']

    from edattr.endorse import StandardEndorsement
    StEnd = StandardEndorsement()
    StEnd.mode = kwargs['endorsement_mode']
    StEnd.kwargs = kwargs

    BESTVALWHERE_DIR = os.path.join(projectdir, 'test_result', 'bestvalwhere.json')
    with open(BESTVALWHERE_DIR) as f:
        best_values_where = json.load(f)["best_values_where"]

    endorsement_samples_by_model = {}
    for mtype, bv in best_values_where.items():
        branch = bv[0] # like k-th fold
        # print(mtype, bv[1], branch)

        TRAIN_VAL_RESULT_DIR = os.path.join(projectdir,'trainval_result',
            f'trainval-output.{prefix}-{branch}.data')
        # results = {
        #     'model' : self.model,
        #     'component': self.components,
        #     'losses': {
        #         'train': losses,
        #         'val' : val_losses, 
        #         },
        #     'confusion_matrices_by_epoch': conf_matrices,
        #     'ntrain': trainloader.dataset.__len__(),
        # }       
        results = joblib.load(TRAIN_VAL_RESULT_DIR)
        model = results['model']

        from accelerate import Accelerator
        accelerator = Accelerator()
        model, dataloader = accelerator.prepare(model, dataloader)
        
        kwargs['description'] = f'best.{"%-10s"%(mtype)} | {prefix}:{branch}' # for progress logging
        e_batch = StEnd.endorse_batchwise(model, dataloader, **kwargs)

        endorsement_samples_by_model[f'best.{mtype}'] = e_batch

        # translate the endorsement in terms of humanly understandable features
        for idx, ends in e_batch.items():
            endo = ends['endorsement']
            new_endo = {}
            for a,b in ends['endorsement'].items():
                # a is possibly an int64, to be converted to int for json compatibility
                new_endo[features[int(a)]] = b 
            e_batch[idx]['endorsement'] = new_endo

            # convert the following to int for json compatibility 
            e_batch[idx]['isCorrect'] = int(e_batch[idx]['isCorrect']) 

    ENDORSEMENT_SAMPLE_DIR = os.path.join(projectdir, 'endorsement.result', 'endorsement_samples.json')
    with open(ENDORSEMENT_SAMPLE_DIR,'w') as f:
        json.dump(endorsement_samples_by_model, f, indent=4)
    print(f'samples saved to {ENDORSEMENT_SAMPLE_DIR}')


import textwrap, re
def fwrap(x):
    x = textwrap.fill(x.get_text(), 21)
    if len(x)>37:
        x = x[:37] +'...'
    return x 

def aggregate_endorsement_samples_vis(**kwargs):
    from edattr.factory import manage_dirs
    DIRS = manage_dirs(**kwargs)  

    PROJECT_PATHS = glob.glob(os.path.join(DIRS['PROJECT_DIR'],
        f'{kwargs["label_suffix"]}_*') )
    PROJECT_LABELS = [os.path.basename(x) for x in PROJECT_PATHS]    
    for projectdir in PROJECT_PATHS:    
        ENDORSEMENT_SAMPLE_DIR = os.path.join(projectdir, 'endorsement.result', 'endorsement_samples.json')

        ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR = os.path.join(projectdir, 'endorsement.visual', 'endorsement_samples')
        if os.path.exists(ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR): 
            shutil.rmtree(ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR) # just resetting
        os.makedirs(ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR, exist_ok=True)

        with open(ENDORSEMENT_SAMPLE_DIR) as f:
            e_samples = json.load(f)

        """
        {'best.acc': {
            '1': {'endorsement': {'feature3': 2, 'feature5': 2}, 'isCorrect': 1, 'y0': 0}, 
            '3': {'endorsement': {'feature4': 2, 'feature3': 2}, 'isCorrect': 1, 'y0': 0}, 
            '5': {'endorsement': {'feature4': 2, 'feature3': 2}, 'isCorrect': 1, ...},}
        'best.X': {...}}         
        """

        max_endo = 0
        for mtype, end_samples in e_samples.items():
            for idx, ends in end_samples.items():
                for feature, endo_value in ends['endorsement'].items():
                    if endo_value> max_endo : max_endo = endo_value

        for mtype, end_samples in e_samples.items():
            for idx, ends in end_samples.items():
                y0 = ends['y0']
                endorsement = ends['endorsement'] 
                isCorrect = ends['isCorrect'] # int 0 or 1
                save_one_endorsement_plot(mtype, idx, y0, endorsement, isCorrect, max_endo, ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR)

def save_one_endorsement_plot(mtype, idx, y0, endorsement, isCorrect, max_endo, ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR):
    if bool(isCorrect):
        correctlabel = 'correct'  
        barcol = (0,0.48,1.0)
    else: 
        correctlabel = 'wrong'
        barcol = (1,0,0.31)

    label = f'{mtype}-{correctlabel}-pred-{str(y0)}'
    IMG_DIR = os.path.join(ENDORSEMENT_SAMPLE_FOLDER_VIS_DIR,
        f'{label}-i{idx}.png')

    x, x_labels = [], []
    for feature, endo_value in endorsement.items():
        x.append(endo_value)
        x_labels.append(feature)
    x_labels = [re.sub('_',' ',f) for f in x_labels]

    font = {'size': 17}
    plt.rc('font', **font)

    plt.figure()
    plt.gcf().add_subplot(111)
    plt.gca().barh(range(len(x)),x[::-1],tick_label=x_labels[::-1], color=barcol, height=0.4)
    plt.gca().set_ylim([-1,len(x)])
    plt.gca().set_xticks(np.arange(0, max_endo+0.2, 1.0))
    plt.gca().set_yticklabels(map(fwrap, plt.gca().get_yticklabels()))
    plt.gca().set_title(label, fontsize=17)
    plt.tight_layout()
    plt.savefig(IMG_DIR)
    plt.close()