from .utils import *
from .decorator import *

def get_home_path():
    if "HOMEPATH" in os.environ: # Windows
        HOME_ = os.environ["HOMEPATH"]
    elif "HOME" in os.environ:
        HOME_ = os.environ["HOME"] # Linux
    else:
        raise Exception('please check how to access your OS home path')    
    return HOME_

def manage_dirs(**kwargs):
    ROOT_DIR = os.getcwd()

    ####### workspace #######
    # Ideally, put all input data and output data here
    #
    WORKSPACE_DIR = kwargs['WORKSPACE_DIR']
    if WORKSPACE_DIR is None:
        HOME_ = get_home_path()
        WORKSPACE_DIR =  os.path.join(HOME_, "Desktop", "edattr.ws") 
        # we put it on the Desktop for no particular reason
    if not os.path.exists(WORKSPACE_DIR):
        print(f'Setting up workspace at {WORKSPACE_DIR}')
    else:
        print(f'Current workspace: {WORKSPACE_DIR}')
    os.makedirs(WORKSPACE_DIR,exist_ok=True)


    ####### results #######
    CKPT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR,exist_ok=True)

    ####### basic directory #######
    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'WORKSPACE_DIR': WORKSPACE_DIR, 
        'CKPT_DIR': CKPT_DIR,   
    }

    if 'DIRECTORY_MODE' not in kwargs:
        return manage_dirs_kfold(DIRS, **kwargs)
    elif kwargs['DIRECTORY_MODE'] == 'singlefile':
        DIRS = manage_dirs_singlefile(DIRS, **kwargs)
    elif kwargs['DIRECTORY_MODE'] == 'bypass':
        pass
    else:
        raise NotImplementedError('Manage dirs mode?')

    return DIRS

def manage_sub_dir_single_data_filename(DIRS, **kwargs):
    ROOT_DIR = DIRS['ROOT_DIR']
    WORKSPACE_DIR = DIRS['WORKSPACE_DIR']
    CKPT_DIR = DIRS['CKPT_DIR']

    DATA_FILE_NAME = kwargs['DATA_FILE_NAME']
    DATA_FOLDER_DIR = os.path.join(WORKSPACE_DIR, 'data')
    os.makedirs(DATA_FOLDER_DIR,exist_ok=True)
    DATA_DIR = kwargs['DATA_DIR']
    if DATA_DIR is None:
        DATA_DIR = os.path.join(DATA_FOLDER_DIR, DATA_FILE_NAME)
    if not os.path.exists(DATA_DIR):
        print(f"No data is found at {DATA_DIR}.\n\n***Consider putting your data file {DATA_FILE_NAME} inside {DATA_FOLDER_DIR}. Or change DATA_DIR to the data file's path.\n\n")
        exit()

    DATA_PROCESSING_DIR = kwargs['DATA_PROCESSING_DIR']
    if DATA_PROCESSING_DIR is None:
        DATA_PROCESSING_DIR = os.path.join(DATA_FOLDER_DIR, DATA_FILE_NAME + '.processing')
    os.makedirs(DATA_PROCESSING_DIR,exist_ok=True)

    DATA_CACHE_DIR = os.path.join(DATA_PROCESSING_DIR, 'data.cache')
    DATA_VIS_DIR = os.path.join(DATA_PROCESSING_DIR, 'datavis') # folder
    os.makedirs(DATA_VIS_DIR,exist_ok=True)

    DIRS.update({
        'DATA_FOLDER_DIR': DATA_FOLDER_DIR,
        'DATA_DIR': DATA_DIR,
        'DATA_PROCESSING_DIR': DATA_PROCESSING_DIR,
        'DATA_CACHE_DIR': DATA_CACHE_DIR,
        'DATA_VIS_DIR': DATA_VIS_DIR, 
    })    
    return DIRS

def manage_sub_dir_project_dir(DIRS, **kwargs):
    CKPT_DIR = DIRS['CKPT_DIR']
    PROJECT_DIR = os.path.join(CKPT_DIR, kwargs['full_projectname'])
    os.makedirs(PROJECT_DIR,exist_ok=True)
    PROJECT_AGGREGATE_DIR = os.path.join(PROJECT_DIR, f'{kwargs["full_projectname"]}-kfold-aggregate.csv')
    DIRS.update({
        'PROJECT_DIR': PROJECT_DIR,   
        'PROJECT_AGGREGATE_DIR': PROJECT_AGGREGATE_DIR,  
    })
    return DIRS

def manage_dirs_singlefile(DIRS, **kwargs):
    ROOT_DIR = DIRS['ROOT_DIR']
    WORKSPACE_DIR = DIRS['WORKSPACE_DIR']
    CKPT_DIR = DIRS['CKPT_DIR']

    DIRS = manage_sub_dir_single_data_filename(DIRS, **kwargs)
    DIRS = manage_sub_dir_project_dir(DIRS, **kwargs)

    if not kwargs['label'] is None:
        PROJECT_DIR = DIRS['PROJECT_DIR']
        LABEL_DIR = os.path.join(PROJECT_DIR, kwargs['label'])
        os.makedirs(LABEL_DIR,exist_ok=True)
        TRAINVAL_RESULT_DIR = os.path.join(LABEL_DIR,'trainval_result')
        TRAINVAL_RESULT_IMG_DIR = os.path.join(TRAINVAL_RESULT_DIR,'imgs')
        os.makedirs(TRAINVAL_RESULT_IMG_DIR, exist_ok=True)
        BEST_MODELS_DIR = os.path.join(LABEL_DIR,'best_models')
        os.makedirs(BEST_MODELS_DIR, exist_ok=True)
        TEST_RESULT_DIR = os.path.join(LABEL_DIR,'test_result')
        os.makedirs(TEST_RESULT_DIR, exist_ok=True)

        DATA_STANDARD_INDICES_DIR = os.path.join(LABEL_DIR, 'standard_indices.data')

        TRAIN_VAL_LOG_DIR = os.path.join(TRAINVAL_RESULT_DIR, 'trainval_log.txt')
        TEST_LOG_DIR = os.path.join(TEST_RESULT_DIR, 'test_log.txt')    

        ENDORSE_RESULT_DIR = os.path.join(LABEL_DIR,'endorsement.result')
        os.makedirs(ENDORSE_RESULT_DIR, exist_ok=True)
        ENDORSE_VIS_DIR = os.path.join(LABEL_DIR,'endorsement.visual')
        os.makedirs(ENDORSE_VIS_DIR, exist_ok=True)   

        EEC_RESULT_DIR = os.path.join(LABEL_DIR,'eec.result')
        os.makedirs(EEC_RESULT_DIR, exist_ok=True)      
        EEC_VIS_DIR = os.path.join(LABEL_DIR,'eec.visual')
        os.makedirs(EEC_VIS_DIR, exist_ok=True)   

        DIRS.update({
            'LABEL_DIR': LABEL_DIR,
            'TRAINVAL_RESULT_DIR': TRAINVAL_RESULT_DIR,
            'TRAINVAL_RESULT_IMG_DIR':TRAINVAL_RESULT_IMG_DIR,
            'BEST_MODELS_DIR': BEST_MODELS_DIR,
            'TEST_RESULT_DIR':TEST_RESULT_DIR,
            'DATA_STANDARD_INDICES_DIR':DATA_STANDARD_INDICES_DIR,
            'TRAIN_VAL_LOG_DIR': TRAIN_VAL_LOG_DIR,
            'TEST_LOG_DIR': TEST_LOG_DIR,  
            'ENDORSE_RESULT_DIR': ENDORSE_RESULT_DIR,
            'ENDORSE_VIS_DIR': ENDORSE_VIS_DIR,
            'EEC_RESULT_DIR': EEC_RESULT_DIR,
            'EEC_VIS_DIR': EEC_VIS_DIR,
            })
        
    return DIRS

def manage_dirs_kfold(DIRS, **kwargs):
    ROOT_DIR = DIRS['ROOT_DIR']
    WORKSPACE_DIR = DIRS['WORKSPACE_DIR']
    CKPT_DIR = DIRS['CKPT_DIR']

    DIRS = manage_sub_dir_single_data_filename(DIRS, **kwargs)
    DIRS = manage_sub_dir_project_dir(DIRS, **kwargs)

    if not kwargs['label'] is None:
        PROJECT_DIR = DIRS['PROJECT_DIR']
        LABEL_DIR = os.path.join(PROJECT_DIR, kwargs['label'])
        os.makedirs(LABEL_DIR,exist_ok=True)
        TRAINVAL_RESULT_DIR = os.path.join(LABEL_DIR,'trainval_result')
        TRAINVAL_RESULT_IMG_DIR = os.path.join(TRAINVAL_RESULT_DIR,'imgs')
        os.makedirs(TRAINVAL_RESULT_IMG_DIR, exist_ok=True)
        BEST_MODELS_DIR = os.path.join(LABEL_DIR,'best_models')
        os.makedirs(BEST_MODELS_DIR, exist_ok=True)
        TEST_RESULT_DIR = os.path.join(LABEL_DIR,'test_result')
        os.makedirs(TEST_RESULT_DIR, exist_ok=True)

        DATA_KFOLD_INDICES_DIR = os.path.join(LABEL_DIR, 'kfold_indices.data')

        TRAIN_VAL_LOG_DIR = os.path.join(TRAINVAL_RESULT_DIR, 'trainval_log.txt')
        TEST_LOG_DIR = os.path.join(TEST_RESULT_DIR, 'test_log.txt')    

        ENDORSE_RESULT_DIR = os.path.join(LABEL_DIR,'endorsement.result')
        os.makedirs(ENDORSE_RESULT_DIR, exist_ok=True)
        ENDORSE_VIS_DIR = os.path.join(LABEL_DIR,'endorsement.visual')
        os.makedirs(ENDORSE_VIS_DIR, exist_ok=True)   

        EEC_RESULT_DIR = os.path.join(LABEL_DIR,'eec.result')
        os.makedirs(EEC_RESULT_DIR, exist_ok=True)      
        EEC_VIS_DIR = os.path.join(LABEL_DIR,'eec.visual')
        os.makedirs(EEC_VIS_DIR, exist_ok=True)   

        DIRS.update({
            'LABEL_DIR': LABEL_DIR,
            'TRAINVAL_RESULT_DIR': TRAINVAL_RESULT_DIR,
            'TRAINVAL_RESULT_IMG_DIR':TRAINVAL_RESULT_IMG_DIR,
            'BEST_MODELS_DIR': BEST_MODELS_DIR,
            'TEST_RESULT_DIR':TEST_RESULT_DIR,
            'DATA_KFOLD_INDICES_DIR':DATA_KFOLD_INDICES_DIR,
            'TRAIN_VAL_LOG_DIR': TRAIN_VAL_LOG_DIR,
            'TEST_LOG_DIR': TEST_LOG_DIR,  
            'ENDORSE_RESULT_DIR': ENDORSE_RESULT_DIR,
            'ENDORSE_VIS_DIR': ENDORSE_VIS_DIR,
            'EEC_RESULT_DIR': EEC_RESULT_DIR,
            'EEC_VIS_DIR': EEC_VIS_DIR,
            })
    return DIRS

def clean_up_directory(**kwargs):
    # warning! This delete the whole project with the given label
    DIRS = manage_dirs(**kwargs)
    shutil.rmtree(DIRS['LABEL_DIR'])
    print(f"Removed unused folder {DIRS['LABEL_DIR']}")

############################
#    Some utils
############################

def plot_losses(train_loss, val_loss, LOSS_PLOT_DIR, avg_every_n=16):
    n_train_loss = len(train_loss)
    n_val_loss = len(val_loss)
    iters = np.arange(n_train_loss)
    iters_val = np.linspace(iters[0],iters[-1],n_val_loss)

    font = {'size': 16}
    plt.rc('font', **font)

    plt.figure()
    n_every = avg_every_n
    iters1, train_loss1= average_every_n(train_loss,iters=iters, n=n_every)
    plt.gca().plot(iters1, train_loss1, c='b',label='train')
    plt.gca().plot(iters, train_loss, c='b', alpha=0.3)

    iters_val1_, val_loss1 = average_every_n(val_loss,iters=iters_val, n=n_every)
    iters_val1 = np.linspace(iters1[0], iters1[-1], len(iters_val1_))
    plt.gca().plot(iters_val1, val_loss1 , c='goldenrod', label='val*')
    plt.gca().plot(iters_val, val_loss, c='gold', alpha=0.3)
    plt.gca().set_xlabel("iters\n*stretched along horizontal-axis")
    plt.gca().set_ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_DIR)
    plt.close()    

def plot_confusion_matrices_by_epoch(metric_types, cm_by_epoch, CM_EPOCH_DIR):        
    vals = {m:[] for m in metric_types}
    for cm in cm_by_epoch:
        for m in metric_types:
            vals[m].append(cm[m])
    
    font = {'size': 16}
    plt.rc('font', **font)
    plt.figure()
    for m in vals:
        plt.plot(vals[m], label=m)
    plt.gca().set_xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CM_EPOCH_DIR)
    plt.close()


############################
#    Pipeline Objects
############################

class AssemblyLine():
    def __init__(self, DIRS, **kwargs):
        super(AssemblyLine, self).__init__()
        self.DIRS = DIRS
        self.kwargs = kwargs    

        self.set_config()
        self.set_dataset()

    def set_config(self):
        raise NotImplementedError( UMSG_IMPLEMENT_DOWNSTREAM )

    def set_dataset(self):
        raise NotImplementedError( UMSG_IMPLEMENT_DOWNSTREAM )

    @staticmethod
    def get_tqdm_progress_bar(iterator, n, disable_tqdm=False):
        # n is the size of iterator
        progress = tqdm.tqdm(iterator, total=n,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
            disable=disable_tqdm)
        return progress


####### Decorators #######
def best_update_log(func):
    def decorated_func(*args,**kwargs):
        func_output = func(*args, **kwargs)

        if 'tv_log' in kwargs:
            kwargs['tv_log'].write(func_output)

        return func_output
    return decorated_func


def test_result_log(func):
    def decorated_func(*args,**kwargs):
        func_output = func(*args, **kwargs)

        log_txt = f'test results on models with best <metric_type>:\n'
        if "k" in kwargs:
            log_txt = f'k={kwargs["k"]} ' + log_txt

        kwargs['test_log'].write(log_txt)
        if 'n_params' in kwargs:
            kwargs['test_log'].write(f'n params:{kwargs["n_params"]}\n')
        if 'test_log' in kwargs:
            kwargs['test_log'].write(json.dumps(func_output, indent=2))
            kwargs['test_log'].write('\n')

        return func_output
    return decorated_func


def get_timer_decorator(func, progress, txt=''):
    def dec_func(*args, **kwargs):  
        enable = False   
        if 'enable_timer' in kwargs:
            if kwargs['enable_timer']: enable = True
            
        if enable: start = time.time()
        f_out = func(*args, **kwargs)
        if enable:
            end = time.time()
            elapsed = end - start        
            
            progress_txt = 'epoch=%s time taken %s[s] = %s [min]'%(
                str(kwargs['epoch']),str(round(elapsed,1)), str(round(elapsed/60.,1)), )
            if "k" in kwargs:
                progress_txt = f"k={kwargs['k']} " + progress_txt 

            progress_txt = txt + progress_txt
            progress.write(progress_txt)
        return f_out    
    return dec_func


############################
#         Standard
############################

class StandardAssemblyLine(AssemblyLine):
    def __init__(self, DIRS, **kwargs):
        self.dataset = None

        super(StandardAssemblyLine, self).__init__(DIRS, **kwargs)
        self.model = None
        self.components = None

        self.set_model()
        self.set_components()

    def set_dataset(self, **kwargs):
        # set self.dataset
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def get_dataloader(self, **kwargs):
        # get pytorch DataLoader from dataset
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def set_model(self, **kwargs):
        self.model = self.init_new_model(**kwargs)

    def init_new_model(self, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def log_model_number_of_params(self, **kwargs):
        nparam = count_parameters(self.model)
        model_type = str(type(self.model))
        SIMPLE_LOG_DIR = os.path.join(self.DIRS['PROJECT_DIR'], 's_log.json')
        s_log = {}
        if os.path.exists(SIMPLE_LOG_DIR):
            with open(SIMPLE_LOG_DIR) as f:
                s_log = json.load(f)

        s_log.update({kwargs['label']: {'model_type':model_type,'nparam':nparam}})
        with open(SIMPLE_LOG_DIR, 'w') as json_file:
            json.dump(s_log, json_file, indent=4)
        print('SIMPLE_LOG_DIR:',SIMPLE_LOG_DIR)


    def set_components(self, **kwargs):
        self.components = self.init_new_components(**kwargs)

    def init_new_components(self, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

class StandardAssemblyLineClassifier(StandardAssemblyLine):
    def __init__(self, DIRS, **kwargs):
        super(StandardAssemblyLineClassifier, self).__init__(DIRS, **kwargs)
        self.get_metric()
        self.best_values = {}
        self.best_values_where = {}

    def get_metric(self):
        from .metric import compute_classification_metrics
        self.compute_metrics = compute_classification_metrics

    #################  Main process #################
    @printoutput
    def train_val_test(self, verbose=0):  
        self.train_val(verbose=verbose)
        self.test(verbose=verbose)

        self.store_best_value_where()
        STATUS = f"The results are stored in folder {self.DIRS['LABEL_DIR']}"
        return STATUS

    def store_best_value_where(self):
        BEST_VALUE_WHERE_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'bestvalwhere.json')
        tmp = {
            '_format_':  {'metric_type':('-BRANCH-', 'metric_value')}, # for ease of reading
            'best_values_where': self.best_values_where
        }
        with open(BEST_VALUE_WHERE_DIR, 'w') as json_file:
            json.dump(tmp, json_file, indent=2, sort_keys=True)

    def train_val(self, verbose=0, **kwargs): 
        trainloader = self.get_dataloader(split='train', shuffle=True)        
        valloader = self.get_dataloader(split='val', shuffle=True)

        results = self.train_val_(trainloader, valloader, verbose=verbose)

        # """ !! save.result !!
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
        # """
        # kth_fold_results = self.train_val_kth_fold_(k, trainloader, valloader, verbose=verbose)

        # like .../trainval_result/trainval-output.branch-main.data
        RESULT_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_DIR'], 
            "trainval-output.branch-main.data")
        joblib.dump(results, RESULT_DIR)

    def train_val_(self, trainloader, valloader, **kwargs):
        if 'verbose' in self.kwargs:
            disable_tqdm=True if self.kwargs['verbose']<=0 else False
        else:
            disable_tqdm=False # let's make default a bit verbose

        ####### EARLY_STOPPING #######
        # let's force user to use EARLY_STOPPING
        assert('early_stopping' in self.config)
        es_conf = self.config['early_stopping']

        assert(es_conf['val_every_n_iters']>=len(valloader))
        # Why? We don't want to validate "too often". Furthermore, this helps us compare 
        # loss plot more sensibly; see for example EECVisualization plot_losses_compare() )     

        ####### Accelerator #######
        from accelerate import Accelerator
        accelerator = Accelerator()
        self.model, self.components['optimizer'], self.components['scheduler'], trainloader, valloader, self.criterion = \
            accelerator.prepare(
                self.model, self.components['optimizer'], self.components['scheduler'], trainloader, valloader, self.criterion
            )

        n_epochs = self.config['n_epochs']
        
        ####### Validation pipeline #######
        losses = []
        val_losses = []
        def val_one_epoch(**kwargs):
            self.model.eval()
            pred_, y0_ = [], []

            for i,(idx, x,y0) in enumerate(valloader):
                # progress.set_description(f'  val epoch:{"%-3s"%(str(epoch))}')
                y = self.model(x.to(torch.float))
                loss = self.criterion(y,y0)
                val_losses.append(loss.item())

                pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
                y0_.extend(y0.cpu().detach().numpy())

            confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
            return confusion_matrix

        TRAIN_VAL_LOG_DIR = self.DIRS['TRAIN_VAL_LOG_DIR']
        tv_log = open(TRAIN_VAL_LOG_DIR,'a')

        conf_matrices = []
        def val_pipeline(epoch):
            with torch.no_grad():
                confusion_matrix = val_one_epoch(epoch=epoch)
                conf_matrices.append(confusion_matrix)

                self.update_best_models_tracker(epoch, confusion_matrix, tv_log=tv_log)
                self.ship_out_best_models(**kwargs)
            return confusion_matrix


        ####### Training here #######
        ES_SIGNAL = False
        globalcounter = 0
        for epoch in range(n_epochs):            
            progress = tqdm.tqdm(enumerate(trainloader), 
                total=len(trainloader),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
            ) 

            self.model.train()
            for i,(idx, x,y0) in progress:
                progress.set_description(f'train epoch:{"%-3s"%(str(epoch))}')
                self.components['optimizer'].zero_grad()
                y = self.model(x.to(torch.float))
                loss = self.criterion(y,y0)
                losses.append(loss.item())                
                accelerator.backward(loss)

                self.components['optimizer'].step()

                ####### Early STOPPING part #######
                if globalcounter % es_conf['val_every_n_iters'] > 0: 
                    globalcounter+=1; continue
                confusion_matrix = val_pipeline(epoch)
                self.components['scheduler'].step()
                
                if globalcounter < es_conf['min_train_iters']: 
                    globalcounter+=1; continue
                ES_SIGNAL = self.early_stopper(confusion_matrix,es_conf['metrics_target'])
                globalcounter+=1
                if ES_SIGNAL: break
            
            if ES_SIGNAL:
                print('>>>>>>> Early Stopping SIGNAL triggered. Great, target reached!') 
                break

            

        tv_log.close()

        results = {
            'model' : self.model,
            'component': self.components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_val_iter': conf_matrices,
            'ntrain': trainloader.dataset.__len__(), # no. of training data points
        }
        return results

    @staticmethod
    def early_stopper(confusion_matrix, metrics_target):
        ES_SIGNAL = True
        for metric, target in metrics_target.items():
            # print(f'{metric}:{confusion_matrix[metric]}')
            if confusion_matrix[metric] < target: return False            
        return ES_SIGNAL

    @staticmethod
    def get_timer_option(epoch, **kwargs):
        verbose = kwargs['verbose']
        enable_timer = False
        
        enable_timer = (epoch<3) and verbose>=20
        return enable_timer

    @best_update_log
    def update_best_models_tracker(self, epoch, confusion_matrix, **kwargs):
        if epoch==0:
            self.best_values = ClassifierBVT()

        STATUS = ""
        bvt_ = self.best_values
        for m in self.config['metric_types']:
            best_m_value, _ = bvt_.get_best_value(m)
            if confusion_matrix[m] >= best_m_value:
                # YES! You need deepcopy, else the update function saves references to the evolving model
                bvt_.update_best_value(m, copy.deepcopy(confusion_matrix[m]), epoch)
                bvt_.update_best_model(m, copy.deepcopy(self.model))
                STATUS += f'updating epoch:{epoch}, new best {m} value={round(confusion_matrix[m],3)}\n'
        return STATUS

    def ship_out_best_models(self, **kwargs):
        """ !! save.result !!
        class ClassifierBVT():
            # BVT: Best Value Trackers
            def __init__(self):
                self.acc = {'value': -1, 'epoch': -1, 'model': None}
                self.recall = ...
        """
        bvt = self.best_values

        # like .../best_models/bestval-output.data
        RESULT_BEST_DIR = os.path.join(self.DIRS['BEST_MODELS_DIR'], 
            "bestval-output.data")
        joblib.dump(bvt, RESULT_BEST_DIR)        

    ################# Test #################
    def load_best_models(self):
        RESULT_BEST_DIR = os.path.join(self.DIRS['BEST_MODELS_DIR'], 
            'bestval-output.data')
        
        # see ship_out_best_models()
        bvt = joblib.load(RESULT_BEST_DIR)
        return bvt

    def test(self, verbose=0, **kwargs):
        testloader = self.get_dataloader(split='test', shuffle=False)
        
        """ !! save.result !! """
        test_results = self.test_(testloader, verbose=verbose)

        BRANCH = 'main' 
        """ ======= Convention =======
        No, there is no special meaning in MAIN branch. This is just for notational
        consistency. In k-fold train/val/test, the BRANCH refers to the k-th fold.
        """

        ####### best value where #######
        # this small chunk
        for metric_type in self.config['metric_types']:
            conf_matrix = test_results['best.'+metric_type]
            val = conf_matrix[metric_type]

            if not metric_type in self.best_values_where:
                self.best_values_where[metric_type] = (BRANCH, val)            

            running_val = self.best_values_where[metric_type][1]
            if val >= running_val:
                self.best_values_where[metric_type] = (BRANCH, val)            

        ####### Save results ####### 
        # like ...<labelname>/test_result/test-output.branch-main.data
        TEST_RESULT_DATA_NAME = 'test-output.branch-main.data'
        TEST_RESULT_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], TEST_RESULT_DATA_NAME)
        joblib.dump(test_results, TEST_RESULT_DIR)
        return f"Test results saved as {TEST_RESULT_DATA_NAME}"
        
    def test_(self, testloader, **kwargs):
        bvt = self.load_best_models()

        test_results = {}
        for m in self.config['metric_types']:
            model = bvt.get_best_model(m)            
            test_results['best.'+m] = self.test_model(model, testloader)            

        TEST_LOG_DIR = self.DIRS['TEST_LOG_DIR']
        test_log = open(TEST_LOG_DIR,'a')
        self.view_test_results(test_results, test_log=test_log, n_params=count_parameters(model))
        test_log.close()
        return test_results        

    @test_result_log
    def view_test_results(self, test_results, **kwargs):
        # just a logging function to be wrapped, does nothing else
        return test_results

    def test_model(self, model, testloader): 
        model.eval()

        from accelerate import Accelerator  
        accelerator = Accelerator()
        model, testloader = accelerator.prepare(model, testloader)

        pred_, y0_ = [], []
        for i,(idx, x,y0) in enumerate(testloader):
            y = model(x.to(torch.float))

            pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
            y0_.extend(y0.cpu().detach().numpy())    

            if self.kwargs['DEV_ITER']>0:
                if i>=self.kwargs['DEV_ITER']: break            

        confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
        return confusion_matrix 

    ################# Visualize #################

    def visualize_output(self):
        print('visualize_output (train/val)...')
        TRAIN_VAL_LOG_DIR = self.DIRS['TRAIN_VAL_LOG_DIR']
        tv_log = open(TRAIN_VAL_LOG_DIR,'a')

        TRAINVAL_RESULT_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_DIR'], 
            'trainval-output.branch-main.data')
        results = joblib.load(TRAINVAL_RESULT_DIR) 

        tv_log.write(f'\ntrainval result \n')
        for key, item in results.items():
            if key == 'losses':
                LOSS_PLOT_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_IMG_DIR'], f'losses_trainval.png')
                tv_log.write(f"{key} saved to {LOSS_PLOT_DIR}\n")
                train_loss = item['train']
                val_loss = item['val']
                plot_losses(train_loss, val_loss, LOSS_PLOT_DIR)                    
            elif key == 'confusion_matrices_by_epoch':
                CM_EPOCH_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_IMG_DIR'], f'confusion_matrix_by_epoch.png')
                tv_log.write(f"{key}: saved to {CM_EPOCH_DIR}\n")
                plot_confusion_matrices_by_epoch(self.config['metric_types'] , item, CM_EPOCH_DIR)
            else:
                tv_log.write(f"{key} : {type(item)}\n")
        tv_log.close()
        print('done!')

    ########### Model Selection ##################
    def select_models(self, model_selection, **kwargs):
        # Again, in standard mode (in contrast to kfold), a "branch" is just a formality
        # It's like k in k-th fold, except there is really only 1 branch.
        if model_selection == 'auto':
            # There's no real selection done in this mode
            best_values_where = self.load_best_value_where()
            branches, metric_types_per_branch = [], []
            for metric_type, kval in best_values_where.items():
                branch = kval[0] 
                if branch in branches:
                    idx_ = branches.index(branch) 
                    metric_types_per_branch[idx_].append(metric_type)
                else:
                    branches.append(branch)
                    metric_types_per_branch.append([metric_type])
        else:
            raise NotImplementedError('model selection invalid')
        return branches, metric_types_per_branch

    def load_best_value_where(self):
        BEST_VALUE_WHERE_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'bestvalwhere.json')
        with open(BEST_VALUE_WHERE_DIR) as f:
            best_values_where = json.load(f)["best_values_where"]
        return best_values_where    

############################
#          kFold
############################

class kFoldAssemblyLine(AssemblyLine):
    def __init__(self, DIRS, **kwargs):

        # Set number of folds when you set the dataset
        # this is done during the initiation of its parent class 
        self.dataset = None
        self.number_of_folds = -1

        # parent class AssemblyLine initiate the following:
        # self.set_config()
        # self.set_dataset()
        # 
        # They're abstract and need to be implemented in applications
        super(kFoldAssemblyLine, self).__init__(DIRS, **kwargs)        

        # The following are indexed by k, corresponding to the k-th fold
        self.models = {} 
        self.components = {}

        self.set_models_by_folds()
        self.set_components_by_folds()        

    def set_dataset(self, **kwargs):
        # set self.dataset
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def get_dataloader(self, **kwargs):
        # get pytorch DataLoader from dataset
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def set_models_by_folds(self, **kwargs):
        for k in range(self.number_of_folds):
            self.models[k] = self.init_new_kth_model(k=k, **kwargs)

    def log_model_number_of_params(self, **kwargs):
        nparam = 0
        model_type = None
        for k in range(self.number_of_folds):
            nparam_ = count_parameters(self.models[k])
            model_type_ = str(type(self.models[k]))
            if nparam == 0:
                nparam += nparam_
                model_type = model_type_
            else:
                assert(nparam == nparam_)
                assert(model_type == model_type_)

        SIMPLE_LOG_DIR = os.path.join(self.DIRS['PROJECT_DIR'], 's_log.json')
        s_log = {}
        if os.path.exists(SIMPLE_LOG_DIR):
            with open(SIMPLE_LOG_DIR) as f:
                s_log = json.load(f)

        s_log.update({kwargs['label']: {'model_type':model_type,'nparam':nparam}})
        with open(SIMPLE_LOG_DIR, 'w') as json_file:
            json.dump(s_log, json_file, indent=4)
        print('SIMPLE_LOG_DIR:',SIMPLE_LOG_DIR)

    
    def init_new_kth_model(k=-1, **kwargs):       
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def set_components_by_folds(self, **kwargs):
        for k in range(self.number_of_folds):
            components = self.init_new_kth_components(k=k, **kwargs)
            self.components[k] = components

    def init_new_kth_components(self, k=-1, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    @staticmethod
    def progress_tracker_one_iter(**kwargs):
        k = kwargs['k']
        epoch = kwargs['epoch']
        i = kwargs['iter']
        x = kwargs['x']
        y0 = kwargs['y0'] # ground-truth
        progress = kwargs['progress']

        if k==0:
            if epoch==0 and i==0:
                progress.write(f"Sample data:\n  [i] x.shape | y0")
            if epoch==0 and i<2:
                progress.write(f"  [{i}] {x.shape} | {y0}")
            if epoch==0 and i==2:
                progress.write(f"  ...") 

    def load_best_value_where(self):
        BEST_VALUE_WHERE_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'bestvalwhere.json')
        with open(BEST_VALUE_WHERE_DIR) as f:
            best_values_where = json.load(f)["best_values_where"]
        return best_values_where

class ClassifierBVT():
    # BVT: Best Value Trackers
    def __init__(self):
        self.acc = {'value': -1, 'epoch': -1, 'model': None}
        self.recall = {'value': -1, 'epoch': -1, 'model': None}
        self.precision = {'value': -1, 'epoch': -1, 'model': None}
        self.f1 = {'value': -1, 'epoch': -1, 'model': None}

    def get_best_value(self, m):
        # m: one of the config['metric_types'], like acc, recall, precision, f1
        value = getattr(self,m)['value']
        epoch = getattr(self,m)['epoch']
        return value, epoch

    def get_best_model(self, m):
        return getattr(self,m)['model']

    def update_best_value(self, m, value, epoch):
        getattr(self,m)['value'] = value
        getattr(self,m)['epoch'] = epoch       

    def update_best_model(self, m, model):
        getattr(self,m)['model'] = model    

    def _summarized(self):
        metrics, values, epochs = "", "", ""
        for m in self.config['metric_types']:
            metrics += f"{m}/"
            value, epoch = self.get_best_value(m)
            values += f"{value}/"
            epochs += f"{epoch}/"
        summ = f"{metrics[:-1]}={values[:-1]} at epochs {epoch[:-1]}"
        return summ

class kFoldAssemblyLineClassifier(kFoldAssemblyLine):
    def __init__(self, DIRS, **kwargs):
        super(kFoldAssemblyLineClassifier, self).__init__(DIRS, **kwargs)        
        """
        ASSUMPTIONS:
        We load data with python DataLoader such that 
        x has shape (b,*) 
        y0 has shape (b,), a tensor of integer/long 0,1,2,...,c-1, where c is the number of classes
        where b is batch size
        """
        self.get_metric()
        self.best_values_by_kfold = {
            # k : ClassifierBVT()
        }
        self.best_values_where = {
            # <metric_type>: (k, val)
        }

        # models are initiated through the initiation of kFoldAssemblyLine()

    def get_result_name_by_k(self, k, result_type):
        return f"{result_type}.k-{k}.data"        

    def get_metric(self):
        from .metric import compute_classification_metrics
        self.compute_metrics = compute_classification_metrics

    #################  Main process #################
    @printoutput
    def train_val_test(self, verbose=0):        
        assert(self.number_of_folds>=0) # dataset needs to be initiated first
        for k in range(self.number_of_folds):
            self.train_val_kth_fold(k, verbose=verbose)  
            self.test_kth_fold(k, verbose=verbose)          

        self.store_best_value_where()
        STATUS = f"The results are stored in folder {self.DIRS['LABEL_DIR']}"
        return STATUS

    def store_best_value_where(self):
        BEST_VALUE_WHERE_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'bestvalwhere.json')
        tmp = {
            '_format_': {'metric_type':('kfold', 'metric_value')}, # for ease of reading
            'best_values_where': self.best_values_where
        }
        with open(BEST_VALUE_WHERE_DIR, 'w') as json_file:
            json.dump(tmp, json_file, indent=2, sort_keys=True)

    ################# Train / Val #################
    def train_val_kth_fold(self, k, verbose=0, **kwargs):
        # k denotes the k-th fold

        trainloader = self.get_dataloader(k, split='train')
        valloader = self.get_dataloader(k, split='val', shuffle=True)

        """ !! save.result !!
        kth_fold_results = {
            'model' : self.models[k],
            'component': self.components[k],
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices,
            'ntrain': trainloader.dataset.__len__(),
        }        
        """
        kth_fold_results = self.train_val_kth_fold_(k, trainloader, valloader, verbose=verbose)

        # like .../trainval_result/trainval-output.k-0.data
        RESULT_KFOLD_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_DIR'], 
            self.get_result_name_by_k(k, 'trainval-output'))
        joblib.dump(kth_fold_results, RESULT_KFOLD_DIR)

    def train_val_kth_fold_(self, k, trainloader, valloader, **kwargs):
        # This is our main training and validation pipeline
        # k denotes the k-th fold
        # trainloader, valloader are pytorch DataLoader for training and validation respectively

        disable_tqdm=True if self.kwargs['verbose']<=0 else False

        from accelerate import Accelerator
        accelerator = Accelerator()
        self.models[k], self.components[k]['optimizer'], self.components[k]['scheduler'], trainloader, valloader, self.criterion = \
            accelerator.prepare(
                self.models[k], self.components[k]['optimizer'], self.components[k]['scheduler'], trainloader, valloader, self.criterion
            )

        n_epochs = self.config['n_epochs']
        progress = self.get_tqdm_progress_bar(iterator=range(n_epochs), 
            n=n_epochs, disable_tqdm=disable_tqdm)
        
        losses = []
        def train_kth_fold_one_epoch(**kwargs):
            k = kwargs['k']
            epoch = kwargs['epoch']
            verbose = 0
            if 'verbose' in kwargs: verbose = kwargs['verbose']

            self.models[k].train()
            for i,(idx, x,y0) in enumerate(trainloader):
                self.progress_tracker_one_iter(progress=progress, k=k, epoch=epoch, iter=i, x=x, y0=y0)

                self.components[k]['optimizer'].zero_grad()
                y = self.models[k](x.to(torch.float))
                loss = self.criterion(y,y0)
                losses.append(loss.item())                
                accelerator.backward(loss)

                self.components[k]['optimizer'].step()
            self.components[k]['scheduler'].step()

        val_losses = []
        def val_kth_fold_one_epoch(**kwargs):
            k = kwargs['k']

            self.models[k].eval()
            pred_, y0_ = [], []
            for i,(idx, x,y0) in enumerate(valloader):
                y = self.models[k](x.to(torch.float))
                loss = self.criterion(y,y0)
                val_losses.append(loss.item())

                pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
                y0_.extend(y0.cpu().detach().numpy())

            confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
            return confusion_matrix

        TRAIN_VAL_LOG_DIR = self.DIRS['TRAIN_VAL_LOG_DIR']
        tv_log = open(TRAIN_VAL_LOG_DIR,'a')

        conf_matrices = []
        for epoch in progress:            
            enable_timer = self.get_timer_option(k, epoch, **kwargs)
            progress.set_description('train/val k=%-2s'%(str(k)))
            train_kth_fold_one_epoch(k=k, epoch=epoch, enable_timer=enable_timer, **kwargs)
            with torch.no_grad():
                confusion_matrix = val_kth_fold_one_epoch(k=k, epoch=epoch)
                conf_matrices.append(confusion_matrix)

                self.update_kfold_best_models_tracker(k, epoch, confusion_matrix, tv_log=tv_log)
                self.ship_out_kfold_best_models(k, progress=progress, **kwargs)

        tv_log.close()

        kth_fold_results = {
            'model' : self.models[k],
            'component': self.components[k],
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices,
            'ntrain': trainloader.dataset.__len__(), # no. of training data points
        }
        return kth_fold_results

    @staticmethod
    def get_timer_option(epoch,k=0, **kwargs):
        verbose = kwargs['verbose']
        enable_timer = False
        if k == 0:
            enable_timer = (epoch<3) and verbose>=20
        return enable_timer

    @best_update_log
    def update_kfold_best_models_tracker(self, k, epoch, confusion_matrix, **kwargs):
        if epoch==0:
            self.best_values_by_kfold[k] = ClassifierBVT()

        STATUS = ""
        bvt_k = self.best_values_by_kfold[k]
        for m in self.config['metric_types']:
            best_m_value, _ = bvt_k.get_best_value(m)
            if confusion_matrix[m] >= best_m_value:
                # YES! You need deepcopy, else the update function saves references to the evolving model
                bvt_k.update_best_value(m, copy.deepcopy(confusion_matrix[m]), epoch)
                bvt_k.update_best_model(m, copy.deepcopy(self.models[k]))
                STATUS += f'updating {k}-th fold, epoch:{epoch}, new best {m} value={round(confusion_matrix[m],3)}\n'
        return STATUS

    def ship_out_kfold_best_models(self, k, **kwargs):
        """ !! save.result !!
        class ClassifierBVT():
            # BVT: Best Value Trackers
            def __init__(self):
                self.acc = {'value': -1, 'epoch': -1, 'model': None}
                self.recall = ...
        """
        bvt = self.best_values_by_kfold[k]

        # like .../best_models/bestval-output.k-0.data
        RESULT_KFOLD_BEST_DIR = os.path.join(self.DIRS['BEST_MODELS_DIR'], 
            self.get_result_name_by_k(k, 'bestval-output'))
        joblib.dump(bvt, RESULT_KFOLD_BEST_DIR)

    ################# Test #################
    def load_kfold_best_models(self, k):
        RESULT_KFOLD_BEST_DIR = os.path.join(self.DIRS['BEST_MODELS_DIR'], 
            self.get_result_name_by_k(k, 'bestval-output'))
        
        # see ship_out_kfold_best_models()
        bvt = joblib.load(RESULT_KFOLD_BEST_DIR)
        return bvt

    @printoutput
    def test_kth_fold(self, k, verbose=0, **kwargs):
        testloader = self.get_dataloader(k, split='test', shuffle=False)

        """ !! save.result !!
        Example:  
        test_results = {
            'best.acc': confusion_matrix,
            'best.<metric_type>': confusion_matrix,
            ...
        }
        where confusion_matrix is like {'TP': 96, 'TN': 52, 'FP': 27, 'FN': 30, 'acc': 0.72 'recall': 0.76, 'precision': 0.78, 'f1': 0.77}
        """
        test_results = self.test_kth_fold_(k, testloader, verbose=verbose)

        ####### best value where #######
        # this small chunk
        for metric_type in self.config['metric_types']:
            conf_matrix = test_results['best.'+metric_type]
            val = conf_matrix[metric_type]

            if not metric_type in self.best_values_where:
                self.best_values_where[metric_type] = (k, val)            

            running_val = self.best_values_where[metric_type][1]
            if val >= running_val:
                self.best_values_where[metric_type] = (k, val)            

        ####### Save results ####### 
        # like ...<labelname>/test_result/test-output.k-0.data
        TEST_RESULT_DATA_NAME = self.get_result_name_by_k(k, 'test-output')
        TEST_RESULT_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], TEST_RESULT_DATA_NAME)
        joblib.dump(test_results, TEST_RESULT_DIR)
        return f"Test results saved as {TEST_RESULT_DATA_NAME}"
            
    def test_kth_fold_(self, k, testloader, **kwargs):
        # Testing the best few models from the kth fold 

        bvt = self.load_kfold_best_models(k)

        test_results = {}
        for m in self.config['metric_types']:
            model = bvt.get_best_model(m)            
            test_results['best.'+m] = self.test_model(model, testloader)            

        TEST_LOG_DIR = self.DIRS['TEST_LOG_DIR']
        test_log = open(TEST_LOG_DIR,'a')
        self.view_test_kfold_results(test_results, k=k, test_log=test_log, n_params=count_parameters(model))
        test_log.close()
        return test_results

    @test_result_log
    def view_test_kfold_results(self, test_results, **kwargs):
        # just a logging function to be wrapped, does nothing else
        return test_results

    def test_model(self, model, testloader): 
        model.eval()

        from accelerate import Accelerator  
        accelerator = Accelerator()
        model, testloader = accelerator.prepare(model, testloader)

        pred_, y0_ = [], []
        for i,(idx, x,y0) in enumerate(testloader):
            y = model(x.to(torch.float))

            pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
            y0_.extend(y0.cpu().detach().numpy())    

        confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
        return confusion_matrix 

    ################# Visualize #################

    def visualize_output(self):
        print('visualize_output (train/val)...')
        TRAIN_VAL_LOG_DIR = self.DIRS['TRAIN_VAL_LOG_DIR']
        tv_log = open(TRAIN_VAL_LOG_DIR,'a')

        for k in range(self.config['kfold']):
            TRAINVAL_RESULT_KFOLD_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_DIR'], 
                self.get_result_name_by_k(k, 'trainval-output'))
            kth_fold_results = joblib.load(TRAINVAL_RESULT_KFOLD_DIR) 

            tv_log.write(f'\ntrainval result fold:{k}\n')
            for key, item in kth_fold_results.items():
                if key == 'losses':
                    LOSS_PLOT_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_IMG_DIR'], f'losses_trainval.{k}.png')
                    tv_log.write(f"{key} saved to {LOSS_PLOT_DIR}\n")
                    train_loss = item['train']
                    val_loss = item['val']
                    plot_losses(train_loss, val_loss, LOSS_PLOT_DIR)                    
                elif key == 'confusion_matrices_by_epoch':
                    CM_EPOCH_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_IMG_DIR'], f'confusion_matrix_by_epoch.{k}.png')
                    tv_log.write(f"{key}: saved to {CM_EPOCH_DIR}\n")
                    plot_confusion_matrices_by_epoch(self.config['metric_types'], item, CM_EPOCH_DIR)
                else:
                    tv_log.write(f"{key} : {type(item)}\n")
        tv_log.close()
        print('done!')

    ########### Model Selection ##################
    def select_models(self, model_selection, **kwargs):
        if model_selection == 'auto':
            best_values_where = self.load_best_value_where()
            kfold_list, metric_types_per_kfold = [], []
            for metric_type, kval in best_values_where.items():
                k = kval[0]
                if k in kfold_list:
                    idx_ = kfold_list.index(k) # find where k is in the kfold_list (list)
                    metric_types_per_kfold[idx_].append(metric_type)
                else:
                    kfold_list.append(k)
                    metric_types_per_kfold.append([metric_type])
        elif model_selection == 'manual':
            raise NotImplementedError('Deprecated')
        else:
            raise NotImplementedError('model selection invalid')

        return kfold_list, metric_types_per_kfold

    def load_best_value_where(self):
        BEST_VALUE_WHERE_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'bestvalwhere.json')
        with open(BEST_VALUE_WHERE_DIR) as f:
            best_values_where = json.load(f)["best_values_where"]
        return best_values_where          



