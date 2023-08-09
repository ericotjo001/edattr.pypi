"""
setup_inferface1.py is another layer of abstraction. 
  It's just a kind of template to make life easier.
!! Please read about setup_template.py for more details

Abstracted Skeletal class: methods and properties invoked by this object may 
(1) require downstream implementation
(2) assume implementations from parent class
"""

from .utils import *
from .decorator import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .setup_template import DatasetTypeK1, DataSetupTypeK1, DatasetTypeR1, DataSetupTypeR1, DatasetTypeS2, init_new_template_model
from .endorse import kFoldXAIeClassifierEEC, StandardXAIeClassifierEEC

EEC_TARGET_LABEL_NAME = 'y0' # just for consistency

def check_batch_size_vs_dataset(b, dataset_):
    drop_last = True
    if b > dataset_.__len__(): 
        drop_last = False    
    return drop_last


#############################################
#         EEC implementations
#############################################
class EECExecutive():
    def __init__(self, ):
        super(EECExecutive, self).__init__()
        # Abstracted Skeletal class (see note above)   

    @staticmethod
    def verify_same_label_within_partition(y0, THIS_PARTITION_LABEL):
        if THIS_PARTITION_LABEL is None:
            THIS_PARTITION_LABEL = y0
        else:
            assert(THIS_PARTITION_LABEL==y0)
        
    def build_eec_type_a(self, train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX):
        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + 'type-a' 
        os.makedirs(EECsubtype_SUFFIX, exist_ok=True)
        EEC_DATA_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-data-t{threshold}.csv')

        df = pd.DataFrame({})
        for pkey, indices in partitions.items():
            # pkey    like frozenset({(6, 2), (1, 2), ('y0', 2), (5, 2)})
            # indices like [65, 270, 713, ..., 457], each raw idx_ (compatible to csv file) 
            # recall: the endorsement core will come from the training dataset (split=train) 

            df_op = self.eec_partition_to_eec_data_type_a(indices, train_indices, train_dataset_, threshold)
            df = pd.concat((df,df_op))
        
        n, D = df.shape # D is the dimension of features + 1 (target label)
        columns = [f'f{i}' for i in range(D-1)] + [EEC_TARGET_LABEL_NAME]
        df.reset_index()
        df.columns = columns
        df.to_csv(EEC_DATA_DIR, index=False)

    def eec_partition_to_eec_data_type_a(self, indices, train_indices, train_dataset_, threshold):
        """
        Output: df_one_partition
          df_one_partition features are all columns except last. 
          Last column is for target.
        Note. In this particular implementation, 
        1. the column names of this dataframe won't be assigned their original names  
        2.  we use BisectingKMeans as the EEC method. 
        
        We implement some processes in which the inputs are not only numerics (float) 
        but also tokens (int/long). k-means
        """ 

        THIS_PARTITION_LABEL = None
        df_one_partition = pd.DataFrame({}) 

        nsubset = len(indices)
        USE_KMEANS = nsubset > threshold # only apply KMEANS on large partitions

        if USE_KMEANS: X = []
        for idx_ in indices:
            # i is not raw idx_!
            i =  train_indices.index(idx_)                            
            # x: yes, x is already pre-processed (including yeo-johnson normalization)
            idx_raw ,x,y0 = train_dataset_.__getitem__(i)

            # Do some double-checking, just in case 
            assert(idx_raw==idx_)  
            self.verify_same_label_within_partition(y0, THIS_PARTITION_LABEL)

            if USE_KMEANS:
                X.append(x)
            else:
                # small partitions are unique, so let's include them in the EEC subset
                onerow = pd.DataFrame([x.tolist() + [y0]]) 
                df_one_partition = pd.concat([df_one_partition, onerow])

        if USE_KMEANS:
            # Large partitions will be shrunk down with this method
            # Goal: to achieve a smaller data subset that still work well during training.
            X = np.array(X) # array like (n, D)

            # since KMeans is initiated with random states, sometimes bad initiation leads to error. That's why we give it a few tries.
            MAX_TRIES = 4
            for i in range(MAX_TRIES):
                try:
                    kmeans = BisectingKMeans(init='k-means++', n_clusters=threshold).fit(X) 
                except:
                    if i+1==MAX_TRIES:
                        raise RuntimeError(f'BisectingKMeans failed beyond {MAX_TRIES} tries')

            for x in kmeans.cluster_centers_:
                onerow = pd.DataFrame([x.tolist() + [y0]])
                df_one_partition = pd.concat([df_one_partition,onerow])
        return df_one_partition


    def build_eec_type_b(self, train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX):
        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + 'type-b' 
        os.makedirs(EECsubtype_SUFFIX, exist_ok=True)
        EEC_DATA_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-data-t{threshold}.csv')

        df = pd.DataFrame({})
        for pkey, indices in partitions.items():
            df_op = self.eec_partition_to_eec_data_type_b(indices, train_indices, train_dataset_, threshold)
            df = pd.concat((df,df_op))
        
        n, D = df.shape # D is the dimension of features + 1 (target label)
        columns = [f'f{i}' for i in range(D-1)] + [EEC_TARGET_LABEL_NAME]
        df.columns = columns
        df.reset_index()        
        df.to_csv(EEC_DATA_DIR, index=False)

    def eec_partition_to_eec_data_type_b(self, indices, train_indices, train_dataset_, threshold):
        THIS_PARTITION_LABEL = None
        df_one_partition = pd.DataFrame({})         

        nsubset = len(indices)  
        if nsubset <= threshold:
            return df_one_partition

        X = []
        for idx_ in indices:
            i =  train_indices.index(idx_)                            
            idx_raw ,x,y0 = train_dataset_.__getitem__(i)

            assert(idx_raw==idx_)  
            self.verify_same_label_within_partition(y0, THIS_PARTITION_LABEL)

            onerow = pd.DataFrame([x.tolist() + [y0]]) 
            df_one_partition = pd.concat([df_one_partition, onerow])        
        return df_one_partition 

class PostEECbase():
    def __init__(self, ):
        super(PostEECbase, self).__init__()

    def get_eec_trainloader(self, EEC_DATA_DIR, batch_size=None):
        datasetup = DataSetupTypeR1(EEC_DATA_DIR, EEC_TARGET_LABEL_NAME)
        dataset_ = self.eec_dataset_object(datasetup)
        n_eec_train = dataset_.__len__()

        b = self.config['batch_size'] if batch_size is None else batch_size

        drop_last = True
        if b > n_eec_train: drop_last = False

        eec_trainloader_ = DataLoader(dataset_, batch_size=b, shuffle=True, drop_last=drop_last)
        return eec_trainloader_, n_eec_train  

    def post_eec_train_val_(self, *args, **kwargs):
        raise NotImplementedError('Implement Downstream')

    def post_eec_test_(self, testloader, EEC_RESULT_DIR):
        eec_results = joblib.load(EEC_RESULT_DIR)
        model = eec_results['model'] 
        model.eval()

        from accelerate import Accelerator  
        accelerator = Accelerator()
        model, testloader = accelerator.prepare(model, testloader)

        print('post eec test in progress...', end='')
        pred_, y0_ = [], []
        for i,(idx, x,y0) in enumerate(testloader):
            y = model(x.to(torch.float))

            pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
            y0_.extend(y0.cpu().detach().numpy())    
            if self.kwargs['DEV_ITER']>0:
                if i>=self.kwargs['DEV_ITER']: break

        test_results = {
            'confusion_matrix': self.compute_metrics(np.array(pred_), np.array(y0_))
        }    
        return test_results


class PostEEC_Standard(PostEECbase):
    def __init__(self):
        super(PostEEC_Standard, self).__init__()

    def post_eec_train_val_(self, branch, model_type, threshold, eec_trainloader_, valloader, eec_sub_type, **kwargs):
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

        ic = iClassifier(config=self.config)
        # *** init model and components ***
        # must be the same as init_new_kth_model() so that the results are comparable
        model = ic.init_new_model(**self.config)
        components = ic.init_new_components(model)

        from accelerate import Accelerator
        accelerator = Accelerator()
        model, components['optimizer'], components['scheduler'], eec_trainloader_, valloader, self.criterion = \
            accelerator.prepare( model, components['optimizer'], components['scheduler'], eec_trainloader_, valloader, self.criterion)

        n_epochs = self.config['eec_n_epochs']

        ####### Validation pipeline #######
        losses = []
        val_losses = []
        def val_one_epoch(**kwargs):
            model.eval()
            pred_, y0_ = [], []

            for i,(idx, x,y0) in enumerate(valloader):
                y = model(x.to(torch.float))
                loss = self.criterion(y,y0)
                val_losses.append(loss.item())

                pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
                y0_.extend(y0.cpu().detach().numpy())

            confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
            return confusion_matrix

        conf_matrices = []
        def val_pipeline(epoch):
            with torch.no_grad():
                confusion_matrix = val_one_epoch(epoch=epoch)
                conf_matrices.append(confusion_matrix)
                # Note: for now, we don't select the best model based on validation
                # For this expt, our original aim is to show an improved loss curve
                # [[ Maybe list best attained metrics later ]]
            return confusion_matrix

        ####### Training here #######
        ES_SIGNAL = False
        globalcounter = 0
        progress = tqdm.tqdm(range(n_epochs), 
            total=n_epochs,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
        ) 
        for epoch in progress:                        
            progress.set_description(f'post-eec t{threshold} [{model_type}] [{eec_sub_type}]. train epoch ')    
            model.train()
            for i,(x,y0) in enumerate(eec_trainloader_):              
                components['optimizer'].zero_grad()
                y = model(x.to(torch.float))
                loss = self.criterion(y,y0)
                losses.append(loss.item())                
                accelerator.backward(loss)

                components['optimizer'].step()

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
                print('>>>>>>> Early Stopping SIGNAL triggered! Great, target reached.') 
                break                
            

        if len(conf_matrices)==0:
            raise RuntimeError("Unknown")

        eec_results = {
            'model': model,
            'components': components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_val_iter': conf_matrices,            
        }     
        return eec_results   

class PostEEC_kFold(PostEECbase):
    def __init__(self, ):
        super(PostEEC_kFold, self).__init__()
        # Abstracted Skeletal class (see note above)


    def post_eec_train_val_(self, k, model_type, threshold, eec_trainloader_, valloader,eec_sub_type, **kwargs):
        disable_tqdm=True if self.kwargs['verbose']<=0 else False

        ic = iClassifier(config=self.config)
        # *** init model and components ***
        # must be the same as init_new_kth_model() so that the results are comparable
        model = ic.init_new_model(**self.config)
        components = ic.init_new_components(model)

        from accelerate import Accelerator
        accelerator = Accelerator()
        model, components['optimizer'], components['scheduler'], eec_trainloader_, valloader, self.criterion = \
            accelerator.prepare( model, components['optimizer'], components['scheduler'], eec_trainloader_, valloader, self.criterion)

        n_epochs = self.config['n_epochs']
        progress = self.get_tqdm_progress_bar(iterator=range(n_epochs), 
            n=n_epochs, disable_tqdm=disable_tqdm)

        losses = []
        from .factory import get_timer_decorator
        def _epoch_timer_(func): # a decorator
            return get_timer_decorator(func,progress)

        def train_one_epoch(**kwargs):
            verbose = 0
            if 'verbose' in kwargs: verbose = kwargs['verbose']

            model.train()
            for i,(x,y0) in enumerate(eec_trainloader_):
                components['optimizer'].zero_grad()

                y = model(x.to(torch.float))
                loss = self.criterion(y,y0)
                losses.append(loss.item())                
                accelerator.backward(loss)

                components['optimizer'].step()
            components['scheduler'].step()

        val_losses = []
        def val_one_epoch(**kwargs):
            model.eval()
            pred_, y0_ = [], []
            for i,(idx,x,y0) in enumerate(valloader):
                y = model(x.to(torch.float))
                loss = self.criterion(y,y0)
                val_losses.append(loss.item())

                pred_.extend(torch.argmax(y,dim=1).cpu().detach().numpy())  
                y0_.extend(y0.cpu().detach().numpy())

            confusion_matrix = self.compute_metrics(np.array(pred_), np.array(y0_))
            return confusion_matrix                    

        conf_matrices = []
        for epoch in progress:
            enable_timer = self.get_timer_option(epoch, k=k, **kwargs)
            progress.set_description('train/val en:%-7s k=%-17s'%(str(eec_sub_type) , str(f"{k}-{model_type}-t{threshold}"))) 
            train_one_epoch(k=k, epoch=epoch, enable_timer=enable_timer,  **kwargs)
            with torch.no_grad():
                confusion_matrix = val_one_epoch(k=k, epoch=epoch)
                conf_matrices.append(confusion_matrix)

                # Note: for now, we don't select the best model based on validation
                # For this expt, our original aim is to show an improved loss curve
                # [[ Maybe list best attained metrics later ]]

        eec_results = {
            'model': model,
            'components': components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices,            
        }     
        return eec_results   


class EECVisualization():
    def __init__(self, ):
        super(EECVisualization, self).__init__()
        # Abstracted Skeletal class (see note above)

    @printfunc
    def visualize_post_eec_output(self, model_selection='auto', **kwargs):
        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        for i,branch in enumerate(branches):
            for m in metric_types_per_branch[i]:
                model_type = 'best.'+str(m)        
                for eec_sub_type in self.config['eec_modes']:    
                    self.visualize_post_eec_output_per_branch_per_model(branch, model_type, eec_sub_type, prefix=kwargs['prefix'])
        return 'done visualizing post EEC output'

    def visualize_post_eec_output_per_branch_per_model(self, branch, model_type, eec_sub_type, prefix='branch'):

        pname = f'{prefix}-{branch}.{model_type}.partition'
        EECsubtype_ = pname + '.' + eec_sub_type
        EEC_PARTITION_SUFFIX = os.path.join(self.DIRS['EEC_RESULT_DIR'], pname)            
        EEC_RECIPE_DIR = EEC_PARTITION_SUFFIX + '.eecr'  
        EEC_VIS_SUFFIX = os.path.join(self.DIRS['EEC_VIS_DIR'], pname)

        eec_recipe = joblib.load(EEC_RECIPE_DIR)
        summary = eec_recipe['partitions_summary']
        for i,eec_param in enumerate(summary['quantiles']):
            threshold = int(eec_param['quantile'])
            self.post_eec_vis(branch, model_type, threshold, 
                EEC_PARTITION_SUFFIX, EEC_VIS_SUFFIX, eec_sub_type, prefix=prefix)

    def post_eec_vis(self, branch, model_type, threshold, 
        EEC_PARTITION_SUFFIX, EEC_VIS_SUFFIX, eec_sub_type, prefix='branch'):

        ####### Post EEC training result dir #######
        # EECsubtype_SUFFIX is like k-1.best.f1.partition.type-b
        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 
        EEC_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-t{threshold}.output')
        eec_trainval_result = joblib.load(EEC_RESULT_DIR)

        ####### Original training result #######
        # for comparison 
        TRAINVAL_RESULT_BRANCH_DIR = os.path.join(self.DIRS['TRAINVAL_RESULT_DIR'], 
            f"trainval-output.{prefix}-{branch}.data")
        results = joblib.load(TRAINVAL_RESULT_BRANCH_DIR) 

        """
        both eec_trainval_result and results have the same content:
        {
            'model': model,
            'components': components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices, 
            'ntrain': ntrain,           
        }
        """      

        EECsubtype_VIS_SUFFIX = EEC_VIS_SUFFIX + '.' + eec_sub_type
        os.makedirs(EECsubtype_VIS_SUFFIX, exist_ok=True)

        tlabel = f't{threshold}'
        metric_types = self.config['metric_types']

        cml_result_by_model, cml_eec_result_by_model = self.compare_cml(prefix, branch, model_type, eec_sub_type, tlabel)

        self.plot_metrics_compare(metric_types, results, eec_trainval_result, 
            tlabel, eec_sub_type, EECsubtype_VIS_SUFFIX, 
            cml_result_by_model=cml_result_by_model, 
            cml_eec_result_by_model=cml_eec_result_by_model)
        self.plot_losses_compare(results, eec_trainval_result, 
            tlabel, EECsubtype_VIS_SUFFIX)

    def compare_cml(self, prefix, branch, model_type, eec_sub_type, tlabel):
        COMPARE_CML = False
        if 'compare-common-ml' in self.kwargs:
            if bool(self.kwargs['compare-common-ml']): COMPARE_CML = True

        if COMPARE_CML:
            CML_RESULT_DIR = os.path.join(self.DIRS['TEST_RESULT_DIR'], 'common_ml_results.json')
            CML_EEC_RESULT_DIR = os.path.join(self.DIRS['EEC_RESULT_DIR'], 'common_ml_eec_results.json')

            with open(CML_RESULT_DIR) as f:
                common_ml_results = json.load(f)
                # like {'k-0-best.acc': {'KNeighborsClassifier': {'FN': 0, ..., 'acc':,...}
            with open(CML_EEC_RESULT_DIR) as f:
                common_ml_eec_results = json.load(f)
                # like {'k-0-best.acc': {'type-a': {'t2': {'KNeighborsClassifier': {'FN': 0, ..., 'acc':,...} ...

            cml_key = f'{prefix}-{branch}-{model_type}'
            cml_result_by_model = common_ml_results[cml_key]
            cml_eec_result_by_model = common_ml_eec_results[cml_key][eec_sub_type][tlabel]
        else:        
            cml_result_by_model = None
            cml_eec_result_by_model = None
        return  cml_result_by_model, cml_eec_result_by_model

    @staticmethod
    def plot_metrics_compare(metric_types, results, eec_trainval_result, label, eec_label, SAVE_DIR, cml_result_by_model=None, cml_eec_result_by_model=None):
        raise NotImplementedError('Implement downstream')

    @staticmethod
    def plot_losses_compare(results, eec_trainval_result, label, SAVE_DIR,
        avg_every_n=16):
        losses = results['losses']
        ntrain = results['ntrain']
        post_eec_losses = eec_trainval_result['losses']
        ntrain_eec = eec_trainval_result['ntrain']

        train_loss = losses['train']
        val_loss = losses['val']
        eec_train_loss = post_eec_losses['train']
        eec_val_loss = post_eec_losses['val']        

        n_every = avg_every_n

        font = {'size': 21}
        plt.rc('font', **font)        
        plt.figure(figsize=(7,7))
        plt.gcf().add_subplot(1,1,1)
        def plot_one_set(train_loss, val_loss, n_every, 
            linestyle, linewidth=1, label_train='train', label_val='val'):
            n_train_loss = len(train_loss)
            n_val_loss = len(val_loss)
            iters = np.arange(n_train_loss)
            iters_val = np.linspace(iters[0],iters[-1],n_val_loss)

            iters1, train_loss1= average_every_n(train_loss,iters=iters, n=n_every)
            plt.gca().plot(iters1, train_loss1, c='b', linestyle=linestyle,label=label_train, )
            plt.gca().plot(iters, train_loss, c='b', alpha=0.1)

            iters_val1_, val_loss1 = average_every_n(val_loss,iters=iters_val, n=n_every)
            iters_val1 = np.linspace(iters1[0], iters1[-1], len(iters_val1_))
            plt.gca().plot(iters_val1, val_loss1 , c='goldenrod', linestyle=linestyle, label=label_val, alpha=0.77, linewidth=1.0)
            plt.gca().plot(iters_val, val_loss, c='gold', alpha=0.33)

            n_iters = len(iters)
            return n_iters

        _ = plot_one_set(train_loss, val_loss, n_every, 'dashed', 
            label_train=f'train n={ntrain}', label_val='val')
        n_iters = plot_one_set(eec_train_loss, eec_val_loss, n_every, 'solid', linewidth=1.4, label_train=f'train eec n={ntrain_eec}', label_val='val eec')

        plt.gca().set_xlim([None, int(n_iters*1.05)])
        plt.gca().set_xlabel("Iters\n*vals stretched along horizontal-axis")
        plt.gca().set_ylabel("Loss")

        plt.tight_layout()
        lgnd = plt.legend(prop={'size':14})

        FIG_DIR = os.path.join(SAVE_DIR, label+'-losses_comparison.png')
        plt.savefig(FIG_DIR)
        plt.close()  

    @staticmethod
    def plot_common_ml_comparison(i, iter_length, nm, m, c,
        cml_result_by_model, cml_eec_result_by_model,):      
        # i: the i-th metric  
        # m: metric type
        # nm: no. of metric types -1
        # c: plot object's color reference
        
        # cml_result_by_model, cml_eec_result_by_model like {'KNeighborsClassifier': {'FN': 0, ..., 'acc':,...}}

        MARKERS = ["o","+","v","^",">","<","1"]
        DELTA = iter_length / (1 + len(cml_result_by_model) + len(cml_eec_result_by_model))
        XPOS = i/nm * DELTA * 0.8

        ccml = [ x for x in c]
        ccml[1] = 0.7
        ncml = len(cml_result_by_model)
        for i_cml, (cml_model, cm_) in enumerate(cml_result_by_model.items()):
            if cm_ == -1: continue
            plt.gca().scatter([XPOS], [cm_[m]], linestyle='dashed', label=f'{m} {cml_model}', c=[tuple(ccml)], marker=MARKERS[i_cml%len(MARKERS)], alpha=0.3)
            ccml[1] = ccml[1] - i_cml/ncml*0.6
            XPOS += DELTA

        ccml = [ x for x in c]
        ccml[1] = 0.7
        ncml = len(cml_eec_result_by_model)
        for i_cml, (cml_model, cm_) in enumerate(cml_eec_result_by_model.items()):
            if cm_ == -1: continue
            plt.gca().scatter([XPOS], [cm_[m]], label=f'{m} {cml_model} [eec]', c=[tuple(ccml)], marker=MARKERS[i_cml%len(MARKERS)])
            ccml[1] = ccml[1] - i_cml/ncml*0.6
            XPOS += DELTA


class EECEpochWiseVis(EECVisualization):
    # For pipeline that uses kfold (suitable for smaller dataset)
    # In our kfold pipelines, validations are performed after each epoch

    def __init__(self):
        super(EECEpochWiseVis, self).__init__()
        
    def plot_metrics_compare(self, metric_types, results, eec_trainval_result, label, eec_label, SAVE_DIR, cml_result_by_model=None, cml_eec_result_by_model=None):

        cm_by_epoch = results['confusion_matrices_by_epoch']
        eec_cm_by_epoch = eec_trainval_result['confusion_matrices_by_epoch']

        vals = {m:[] for m in metric_types}
        eec_vals = {m:[] for m in metric_types}
        for cm, eec_cm in zip(cm_by_epoch, eec_cm_by_epoch):
            for m in metric_types:
                vals[m].append(cm[m])
                eec_vals[m].append(eec_cm[m])
        
        font = {'size': 16}
        plt.rc('font', **font)
        plt.figure(figsize=(10,8))
        c = [0.22, 0, 1.0]
        nm = len(metric_types) - 1
        if nm<=0: nm=1
        for i,m in enumerate(metric_types):
            c[0] = c[0] + i/nm*0.77
            c[2] = c[2] - i/nm*0.77
            plt.gca().plot(vals[m], c=tuple(c), label=m, linestyle='dashed', alpha=0.5)
            plt.gca().plot(eec_vals[m], c=tuple(c), label=m+ f' [eec-{eec_label}]', alpha=0.77, linewidth=1.0)

            # ############## COMPARE CML ################              
            if cml_result_by_model is None or cml_eec_result_by_model is None: continue
            self.plot_common_ml_comparison(i, len(vals[m]), nm, m, c,
                cml_result_by_model,cml_eec_result_by_model,)

        plt.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.gca().set_xlabel('Epoch')
        plt.tight_layout()

        FIG_DIR = os.path.join(SAVE_DIR, label+'-cm-epoch.png')
        plt.savefig(FIG_DIR)
        plt.close()

class EECValIterWiseVis(EECVisualization):
    def __init__(self):
        super(EECValIterWiseVis, self).__init__()
                
    
    def plot_metrics_compare(self, metric_types, results, eec_trainval_result, label, eec_label, SAVE_DIR, cml_result_by_model=None, cml_eec_result_by_model=None):
        cm_by_valiter = results['confusion_matrices_by_val_iter']
        eec_cm_by_valiter = eec_trainval_result['confusion_matrices_by_val_iter']

        N_EEC_TOOMANY = False
        if len(eec_cm_by_valiter) >= len(cm_by_valiter):
            print("""It seems like you have trained your models on EEC data subset for too many epochs. In that case, there is no improved time efficiency, which makes the idea of EEC rather pointless""")
            N_EEC_TOOMANY = True

        # rearranging original results just for additional layer of plotting
        vals_rearranged_ = {m:[] for m in metric_types}
        for cm in cm_by_valiter:
            for m in metric_types:
                vals_rearranged_[m].append(cm[m])

        # results comparing original with EEC
        vals = {m:[] for m in metric_types}
        eec_vals = {m:[] for m in metric_types}
        for cm, eec_cm in zip(cm_by_valiter, eec_cm_by_valiter):
            for m in metric_types:
                vals[m].append(cm[m])
                eec_vals[m].append(eec_cm[m])

        font = {'size': 16}
        plt.rc('font', **font)
        plt.figure(figsize=(10,8))
        c = [0.22,0,1.0]
        nm = len(metric_types) - 1
        if nm<=0: nm=1
        for i,m in enumerate(metric_types):
            c[0] = c[0] + i/nm*0.77
            c[2] = c[2] - i/nm*0.77
            plt.gca().plot(vals[m], c=tuple(c), label=m, linestyle='dashed', alpha=0.5)
            plt.gca().plot(eec_vals[m], c=tuple(c), label=m+ f' [eec-{eec_label}]', alpha=0.77, linewidth=1.0)

            # ############## COMPARE CML ################              
            if cml_result_by_model is None or cml_eec_result_by_model is None: continue
            self.plot_common_ml_comparison(i, len(vals[m]), nm, m, c,
                cml_result_by_model,cml_eec_result_by_model,)
        plt.gca().set_xlabel('Val iter unit')
        plt.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5))
        if N_EEC_TOOMANY:
            plt.gca().set_title('Reconsider EEC no. of epochs!!')

        c2 = [0,1.0,0]
        plt.gca().twiny()
        for i,m in enumerate(metric_types):
            c2[1] = c2[1] - i/nm*0.27
            c2[2] = c2[2] + i/nm*0.27            
            plt.gca().plot(vals_rearranged_[m], c=tuple(c2), label=f'{m} - (all)', alpha=0.57)
        plt.gca().tick_params(axis='x', colors='g')
        plt.gca().set_xlabel('Val iter unit (all)', c='g', alpha=0.57)
        plt.legend(prop={'size': 11}, framealpha=0.77, labelcolor='linecolor')
      
        plt.tight_layout()

        FIG_DIR = os.path.join(SAVE_DIR, label+'-cm-valiter.png')
        plt.savefig(FIG_DIR)
        plt.close()
    


#############################################
#          Classifier (bare minimum) 
#############################################
# Classification type of setup, vanilla. No kfold etc. 
# e.g. assume data is clearly separated as train, val, test without randomizing indices.

# ****** Dataset objects ******
# !! Assumptions
# y0: target labels for classification, raw labels from the CSV files. 
# Here, assume labels y0 take the values 0,1,... or C-1. 
#
class DatasetSingleClassifierCSV(DatasetTypeR1):
    def __init__(self, setupTypeR1):
        super(DatasetSingleClassifierCSV, self).__init__(setupTypeR1)

    def __getitem__(self, i):
        x = self.df[i]
        y0 = int(self.df_target[i])
        return x, y0         

# ******* iClassifier: independent Classifier *******
# This is an independent system for managing a classification model 
class iClassifier():
    def __init__(self, config):
        super(iClassifier, self).__init__()
        self.config = config

    def init_new_model(self, **kwargs):
        return init_new_template_model(**self.config)

    def init_new_components(self, model, **kwargs):
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'], betas=(0.5,0.999))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)        
        components = {'optimizer': optimizer,'scheduler': scheduler,}
        return components        

#############################################
#             Classifier 
#############################################
# Just a standard classifier pipeline


# ****** Dataset objects ******
# !! Assumptions
# y0: target labels for classification, raw labels from the CSV files. 
# Here, assume labels y0 take the values 0,1,... or C-1. 
#

class DatasetStandardClassifierCSV(DatasetTypeS2):
    def __init__(self, setupTypeS2, split):        
        super(DatasetStandardClassifierCSV, self).__init__(setupTypeS2, split)

    def __getitem__(self, i):
        """
        "indices" is a variable introduced by our standard train/val/test setup. 
        If there are n total rows in the CSV file,  then self.indices will be a subset of 
          [0,1,...,n-1] that depends on your split (train/val/test)
        """
        idx = self.indices[i] # raw index, straight out of the csv file
        x = self.df[idx] # some processing already applied (e.g. normalized with yeo-johnson)
        y0 = int(self.df_target[idx]) 
        return idx, x,y0   

# ****** The standard Classifier Pipeline ******
#

class StandardClassifier(StandardXAIeClassifierEEC, EECExecutive, EECValIterWiseVis, PostEEC_Standard):
    def __init__(self, DIRS, **kwargs):
        EECExecutive.__init__(self)
        PostEEC_Standard.__init__(self) 
        EECVisualization.__init__(self)        
        self.set_dataset_object()
        super(StandardClassifier, self).__init__(DIRS, **kwargs)

    def init_new_model(self, **kwargs):
        # for convenience, let's use our own template initiation
        return init_new_template_model(**self.config)

    def init_new_components(self, **kwargs):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], betas=(0.5,0.999))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
        
        components = {'optimizer': optimizer,'scheduler': scheduler,}
        return components

    def set_dataset(self):    
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def get_dataset_(self, DatasetClass, split):
        # DatasetClass is a class definition (need to be initialized as an object)         
        return DatasetClass(self.dataset, split)  

    def set_dataset_object(self):
        self.dataset_object = DatasetStandardClassifierCSV
        self.eec_dataset_object = DatasetSingleClassifierCSV

    def get_dataloader(self, split='train', shuffle=True, batch_size=None):
        dataset_ = self.get_dataset_(self.dataset_object, split) 

        b = self.config['batch_size'] if batch_size is None else batch_size
        drop_last = check_batch_size_vs_dataset(b, dataset_)
        loader_ = DataLoader(dataset_, batch_size=b, shuffle=shuffle, drop_last=drop_last)
        return loader_        

    def get_dataloader_reduced_trainset(self, shuffle=True, batch_size=None):   
        split = 'train'
        dataset_ = self.get_dataset_(self.dataset_object, split) 

        # Reduced Training Class Size threshold
        reduced_dataset_ = self.reduce_training_dataset(
            copy.deepcopy(dataset_), 
            RTCS_threshold=self.config['RTCS_threshold'],
            RTCS_mode= self.config['RTCS_mode']
        ) 
        # we use deep copy just to be sure that nothing in the actual dataset_ gets altered

        b = self.config['batch_size'] if batch_size is None else batch_size
        drop_last = check_batch_size_vs_dataset(b, reduced_dataset_)
        loader_ = DataLoader(reduced_dataset_, batch_size=b, shuffle=shuffle, drop_last=drop_last)
        return loader_        

    def reduce_training_dataset(self, dataset_, RTCS_threshold=None, RTCS_mode='absolute', RTCS_max=4096):
        """ RTCS: Reduced Training Class Size 
        if RTCS_mode='absolute', 
           make sure RTCS_threshold is an integer, specifying the absolute number of data points we want per class
        if RTCS_mode='fraction'
            make sure RTCS_threshold is between 0 and 1
        """
        reduced_indices = []
        y0s = list(set(dataset_.df_target))
        df_ = dataset_.df
        n_df = len(df_)
        # print(dataset_.df) # already normalized and tokenized. Target still raw.

        REDUCED_TRAINSET_INDICES_DIR = os.path.join(self.DIRS['ENDORSE_RESULT_DIR'],
            'reduced_trainset_indices.data')
        
        if os.path.exists(REDUCED_TRAINSET_INDICES_DIR):
            print(f'Loading reduced trainset indices at {REDUCED_TRAINSET_INDICES_DIR}\n')
            reduced_indices = joblib.load(REDUCED_TRAINSET_INDICES_DIR)
        else:
            print('Reducing training dataset...')
            FEATURES = list(self.TOKEN_FEATURES) + list(self.NUMERICAL_FEATURES) 
            classwise_data = {}
            classwise_data_idx = {}
            for y0 in y0s:
                subdf = df_[df_[self.TARGET_LABEL_NAME]==y0][FEATURES]
                classwise_data[y0] = subdf
                classwise_data_idx[y0] = subdf.index.tolist()

            progress = tqdm.tqdm(classwise_data, total=len(classwise_data),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for y0 in progress:
                features_ = np.array(classwise_data[y0])
                indices_ = np.array(classwise_data_idx[y0])
                # print(features_.shape) # shape: (n,D)
                # print(indices_.shape) # shape: (n,)

                ######## Determine the number of clusters ########
                n_class = len(df_[df_[self.TARGET_LABEL_NAME]==y0])
                if RTCS_mode == 'absolute':
                    n_clusters = np.min([RTCS_threshold,n_class])
                elif RTCS_mode == 'fraction':
                    n_clusters = int(RTCS_threshold*n_class)
                else:
                    raise NotImplementedError("Unknown RTCS mode")
                n_clusters = np.min([n_clusters, RTCS_max])

                progress.set_description(f'class: {"%-7s"%(str(y0))} n_clusters={n_clusters}  ')

                # a specific implementation of reduction method. You can consider others
                csr_indices, _ , _ = self._kmeans_CSR_(features_, n_clusters=n_clusters)
                reduced_classwise_indices = indices_[csr_indices]
                reduced_indices.extend(reduced_classwise_indices)
            joblib.dump(reduced_indices, REDUCED_TRAINSET_INDICES_DIR)
            print(f'Saving reduced trainset indices to {REDUCED_TRAINSET_INDICES_DIR}\n')
        
        dataset_.indices = reduced_indices
        return dataset_

    @staticmethod
    def compute_similarity(X, indices_, kmeans_labels, cluster_centers_):
        sims = { # similarities
            ULABEL:{'distance': np.inf , 'idx_': None} 
            for ULABEL in set(kmeans_labels)
        }
        for x, idx_, ULABEL in zip(X, indices_, kmeans_labels):    
            center = cluster_centers_[ULABEL,:]

            distance = np.linalg.norm(x-center)

            UPDATE_SCORE = False
            if sims[ULABEL]['idx_'] is None:
                UPDATE_SCORE = True
            elif distance < sims[ULABEL]['distance']: 
                UPDATE_SCORE = True

            if not UPDATE_SCORE: continue

            sims[ULABEL]['idx_'] = idx_
            sims[ULABEL]['distance'] = distance
        return sims

    def _kmeans_CSR_(self,X, n_clusters=1000):
        # CSR: centers similarity reduction
        # Find which vector in X is closest 
        # to the centers found by kmeans
        # Store the indices of these vectors.
        nc = len(X)
        indices_ = range(nc)
        if nc > n_clusters:
            # print('X.shape:', X.shape) # shape (nc, D)

            from sklearn.cluster import BisectingKMeans
            kmeans = BisectingKMeans(n_clusters=n_clusters).fit(X)
            # print(kmeans.cluster_centers_.shape) 
            # shape (RTCS_threshold, D)
            # print(kmeans.labels_) 
            # shape: (nc), list of 0,1,..., RTCS_threshold-1

            sims = self.compute_similarity(X,
                indices_, kmeans.labels_, kmeans.cluster_centers_)

            csr_indices = []
            for _, sim in sims.items():
                csr_indices.append(sim['idx_'])
            return csr_indices, kmeans.cluster_centers_, kmeans.labels_
        else:
            csr_indices = indices_
            return csr_indices, None, None

    def build_endorsement_core_data_subset(self, branch, model_type, eec_param, partitions, EEC_PARTITION_SUFFIX, **kwargs):
        # eec_param is like 
        # {'q': 0.52, 'quantile': 3.9, 'core_fraction': 0.201, 'core_size': 124},
        # Recall (IMPORTANT): partitions only includes training data points that have been correctly predicted. Naturally, this means that its size will be smaller than the size of train dataset (split=train)

        # prepare data for core extraction
        train_dataset_ = self.get_dataset_(self.dataset_object, 'train')

        # unlike the kfold version, we use the reduced train dataset
        # recall: for this pipeline, we assume train dataset is very large
        REDUCED_TRAINSET_INDICES_DIR = os.path.join(self.DIRS['ENDORSE_RESULT_DIR'],
            'reduced_trainset_indices.data')
        assert(os.path.exists(REDUCED_TRAINSET_INDICES_DIR))
        reduced_indices = joblib.load(REDUCED_TRAINSET_INDICES_DIR)
        train_dataset_.indices = reduced_indices
   
        train_indices = list(train_dataset_.indices) # for convenience

        threshold = int(eec_param['quantile'])
        if 'type-a' in self.config['eec_modes']:
            self.build_eec_type_a(train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX) # Method from EECExecutive

        if 'type-b' in self.config['eec_modes']: 
            self.build_eec_type_b(train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX) # Method from EECExecutive

    """ 
    We will implement both post_ecc train/val and test methods here
    See parent classes for more details.
    """            

    def post_eec_train_val(self, branch, model_type, threshold, EEC_PARTITION_SUFFIX, eec_sub_type, **kwargs):
        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 

        EECsubtype_DATA_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-data-t{threshold}.csv')
        eec_trainloader_, n_eec_train = self.get_eec_trainloader(EECsubtype_DATA_DIR)
        valloader = self.get_dataloader(split='val', shuffle=True)

        EECsubtype_results = self.post_eec_train_val_(branch, model_type, threshold, 
            eec_trainloader_, valloader, eec_sub_type, **kwargs)
        EECsubtype_results.update({'ntrain': n_eec_train})

        """ !! save.result !!
        eec_results = {
            'model': model,
            'components': components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices,            
            'ntrain': n_eec_train
        }        
        """
        EEC_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-t{threshold}.output')
        joblib.dump(EECsubtype_results, EEC_RESULT_DIR)

    @printoutput
    def post_eec_test(self, k, model_type, threshold, EEC_PARTITION_SUFFIX, eec_sub_type, **kwargs):
        testloader = self.get_dataloader(split='test', shuffle=False)

        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 
        EEC_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-t{threshold}.output')
        
        """ !! save.result !!
        Here! Only a confusion matrix, though
        """
        test_results = self.post_eec_test_(testloader, EEC_RESULT_DIR)

        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 
        TEST_RESULT_DATA_NAME = f'eec-test-t{threshold}.result'
        EEC_TEST_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, TEST_RESULT_DATA_NAME)
        joblib.dump(test_results, EEC_TEST_RESULT_DIR)
        return f'\rTest results saved as {TEST_RESULT_DATA_NAME}'    

#############################################
#             kFoldClassifier 
#############################################
# Classification type of setup with k-fold validation

# ****** Dataset objects ******
# !! Assumptions
# y0: target labels for classification, raw labels from the CSV files. 
# Here, assume labels y0 take the values 0,1,... or C-1. 
#
class DatasetKFoldClassifierCSV(DatasetTypeK1):
    def __init__(self, setupTypeK1, k, split):         
        # Setting df and df_target up using the parent class (data loaded by setupTypeK1)
        super(DatasetKFoldClassifierCSV, self).__init__(setupTypeK1, k, split)

    def __getitem__(self, i):
        """
        "indices" is a variable introduced by our kfold setup. 
        If there are n total rows in the CSV file,  then self.indices will be a subset of 
          [0,1,...,n-1] that depends on your split (train/val/test)
        """
        idx = self.indices[i] # raw index, straight out of the csv file
        x = self.df[idx] # some processing already applied (e.g. normalized with yeo-johnson)
        y0 = int(self.df_target[idx]) 
        return idx, x,y0   

def get_lr_lambda(epoch):
    # just some default function for learning rate scheduler
    return 0.65 ** epoch

# ****** The kFold Classifier Pipeline ******
#

# The following is an implementation of kFoldClassifier
#   with setup like dataset_object=DatasetKFoldClassifierCSV etc
class kFoldClassifier(kFoldXAIeClassifierEEC, EECExecutive, EECEpochWiseVis, PostEEC_kFold):
    def __init__(self, DIRS, **kwargs):
        EECExecutive.__init__(self)
        PostEEC_kFold.__init__(self) 
        EECEpochWiseVis.__init__(self)
        self.set_dataset_object()
        
        # Parent classes automatically initiate the following: 
        #   self.set_config(), self.set_dataset()
        #   self.models = {...}, self.components = {...}     
        super(kFoldClassifier, self).__init__(DIRS, **kwargs)

    def set_config(self, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)  

    def init_new_kth_model(self, k=-1, **kwargs):
        # k is the k-th fold
        # return a pytorch nn.Module as your model
        # Here, let's use some predefined models
        return init_new_template_model(**self.config)

    def init_new_kth_components(self, k=-1, **kwargs):
        optimizer = optim.Adam(self.models[k].parameters(), lr=self.config['learning_rate'], betas=(0.5,0.999))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
        
        components = {'optimizer': optimizer,'scheduler': scheduler,}
        return components        

    def set_dataset(self):    
        self.number_of_folds = self.config['kfold']
        self.dataset = DataSetupTypeK1(self.DIRS, 
            TARGET_LABEL_NAME=self.TARGET_LABEL_NAME, 
            kfold=self.number_of_folds)  

    def get_dataset_(self, DatasetClass, k, split):
        # DatasetClass is a class definition (need to be initialized as an object)           
        return DatasetClass(self.dataset, k, split)    

    def set_dataset_object(self):
        """ Note: both DatasetKFoldClassifierCSV and DatasetSingleClassifierCSV
        are actually flexible. If your CSV files have some different format, then
        you may need to create your own dataset classes (see projects/fetalhealthclassifier example)
        """
        self.dataset_object = DatasetKFoldClassifierCSV
        self.eec_dataset_object = DatasetSingleClassifierCSV

    def get_dataloader(self, k=-1, split='train', shuffle=True, batch_size=None):
        # we just use a common k-fold validation setup
        
        dataset_ = self.get_dataset_(self.dataset_object, k, split) 

        b = self.config['batch_size'] if batch_size is None else batch_size
        drop_last = check_batch_size_vs_dataset(b, dataset_)
        loader_ = DataLoader(dataset_, batch_size=b, shuffle=shuffle, drop_last=drop_last)
        return loader_

    ##################### EEC ######################
    # The Extraction of Endorsement Cores (EEC) part, see endorse.py

    """ The following functions have been defined in the parent class
    1. def eec_selected_models(**kwargs) which loops through selected models 
       (based on k, metric_type), executing the next function in each loop
    2. def eec_by_model(**kwargs) which loops through eec_param (particularly quantiles), 
       executing build_endorsement_core_data_subset() in each loop
    """

    def build_endorsement_core_data_subset(self, k, model_type, eec_param, partitions, EEC_PARTITION_SUFFIX, **kwargs):
        # eec_param is like 
        # {'q': 0.52, 'quantile': 3.9, 'core_fraction': 0.201, 'core_size': 124},
        # Recall (IMPORTANT): partitions only includes training data points that have been correctly predicted. Naturally, this means that its size will be smaller than the size of train dataset (split=train)

        # prepare data for core extraction
        train_dataset_ = self.get_dataset_(self.dataset_object, k, 'train')
        train_indices = train_dataset_.indices.tolist() # for convenience

        threshold = int(eec_param['quantile'])
        if 'type-a' in self.config['eec_modes']:
            self.build_eec_type_a(train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX) # Method from EECExecutive

        if 'type-b' in self.config['eec_modes']: 
            self.build_eec_type_b(train_indices, train_dataset_, threshold, partitions,  EEC_PARTITION_SUFFIX) # Method from EECExecutive

    """ 
    We will implement both post_ecc train/val and test methods here
    See parent classes for more details.
    """

    def post_eec_train_val(self, k, model_type, threshold, EEC_PARTITION_SUFFIX, eec_sub_type, **kwargs):
        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 

        EECsubtype_DATA_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-data-t{threshold}.csv')
        eec_trainloader_, n_eec_train = self.get_eec_trainloader(EECsubtype_DATA_DIR)
        valloader = self.get_dataloader(k, split='val', shuffle=True)

        EECsubtype_results = self.post_eec_train_val_(k, model_type, threshold, 
            eec_trainloader_, valloader, eec_sub_type, **kwargs)
        EECsubtype_results.update({'ntrain': n_eec_train})

        """ !! save.result !!
        eec_results = {
            'model': model,
            'components': components,
            'losses': {
                'train': losses,
                'val' : val_losses, 
                },
            'confusion_matrices_by_epoch': conf_matrices,            
            'ntrain': n_eec_train
        }        
        """
        EEC_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-t{threshold}.output')
        joblib.dump(EECsubtype_results, EEC_RESULT_DIR)

    @printoutput
    def post_eec_test(self, k, model_type, threshold, EEC_PARTITION_SUFFIX, eec_sub_type, **kwargs):
        testloader = self.get_dataloader(k, split='test', shuffle=False)

        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 
        EEC_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, f'eec-train-t{threshold}.output')
        
        """ !! save.result !!
        Here! Only a confusion matrix, though
        """
        test_results = self.post_eec_test_(testloader, EEC_RESULT_DIR)

        EECsubtype_SUFFIX = EEC_PARTITION_SUFFIX + '.' + eec_sub_type 
        TEST_RESULT_DATA_NAME = f'eec-test-t{threshold}.result'
        EEC_TEST_RESULT_DIR = os.path.join(EECsubtype_SUFFIX, TEST_RESULT_DATA_NAME)
        joblib.dump(test_results, EEC_TEST_RESULT_DIR)
        return f'Test results saved as {TEST_RESULT_DATA_NAME}'

    ##################### Post EEC Visualization ######################
    # See functions implemented by EECVisualization()
