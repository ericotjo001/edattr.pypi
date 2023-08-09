from .utils import *
from .decorator import *
from .factory import kFoldAssemblyLineClassifier, StandardAssemblyLineClassifier

from captum.attr import KernelShap, Lime
from collections import defaultdict

def sort_dictionary_by_values(d):
    dsorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])[::-1]}
    return dsorted

def topn_argmax(x,n=3):
    sorted_ = x[np.argsort(x)]
    topn = sorted_[::-1][:n] # arranged from high to low
    topn_indices = []
    counter = 0

    top_dict = {}
    for thismax in list(set(topn))[::-1]:
        tmp = np.where(x==thismax)
        for idx in tmp[0]:
            if not idx in topn_indices:
                # topn_indices.append(idx)
                top_dict[idx] = x[idx]
                counter += 1
            if counter >= n: break
        if counter >= n: break

    # sorted, decreasing values (not sorted by keys)
    return sort_dictionary_by_values(top_dict) 

def attr_top(attr, n=3):
    # attr: x.shape is (D,)
    topn_indices = topn_argmax(attr, n=n).keys()
    return list(topn_indices)

def edattr_to_feature_attr(endorsement, n_features):
    # endorsement is like {0:2, 2:1, 3:1}
    edattr = np.zeros(shape=n_features)
    for ix, endo in endorsement.items():
        edattr[ix] = endo
    return edattr


class StandardizedAttr():
    """ Important note: this class is kinda arbitrary. But for the sake of consistency and clarity, let's standardize how we use captum methods. 
    """
    def __init__(self):
        super(StandardizedAttr, self).__init__()

        # attr_models is a dynamic variable. It changes, say, with the k-th fold
        self.attr_models = {}

    def populate_attribution_models(self, model):
        if self.mode in ['shap-lime-top2', 'shap-lime-top3']:
            self.attr_models = {
                # we use LIME-based method to compute shap more efficiently
                'kshap' : KernelShap(model), 
                'lime': Lime(model)        
            }
        else:
            raise NotImplementedError()

    def kshap(self, x_, y_pred_, topn=3):
        # both x_, y_pred_ are single data point. Shape is like (D1,D2,...) NOT like (b,D1,D2,...) where b denotes batch size
        attr_kshap = self.attr_models['kshap'].attribute(x_, 
            target=y_pred_.item()) # we use default arguments like n_samples    
        attr_idx_kshap = attr_top(attr_kshap[0].clone().detach().cpu().numpy(), n=topn)
        return attr_kshap, attr_idx_kshap

    def lime(self, x_, y_pred_, topn=3):
        attr_lime = self.attr_models['lime'].attribute(x_, 
            target=y_pred_.item()) # we use default arguments like n_samples
        attr_idx_lime = attr_top(attr_lime[0].clone().detach().cpu().numpy(), n=topn)        
        return attr_lime, attr_idx_lime

class StandardEndorsement(StandardizedAttr):
    """docstring for StandardEndorsement"""
    def __init__(self):
        super(StandardEndorsement, self).__init__()

    @staticmethod
    def endorse(*args):
        endorsement = defaultdict(int)
        for attr_idx_ in args:
            for i in attr_idx_:
                endorsement[i] += 1
        return endorsement

    def endorse_batchwise(self, model, loader, **kwargs):
        # model: pytorch nn.Module
        # loader: pytorch DataLoader
        from accelerate import Accelerator
        accelerator = Accelerator()
        model, loader = accelerator.prepare(model, loader)
        model.eval()

        disable_tqdm=True if kwargs['verbose']==0 else False
        self.populate_attribution_models(model)
        progress = tqdm.tqdm(enumerate(loader), total=len(loader),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
            disable=disable_tqdm,)
        
        e_batch = {}
        for i,(idx, x,y0) in progress:
            progress.set_description(kwargs['description'])
            x = x.to(torch.float)
            y = model(x)

            e_batch_ = self.endorse_batch_(**{ 'data_batch': (idx, x,y0,y), })
            e_batch.update(e_batch_)

            if 'DEV_ITER' in self.kwargs:
                if self.kwargs['DEV_ITER'] > 0:
                    if i>=self.kwargs['DEV_ITER']: break
        return e_batch

    def endorse_batch_(self, **kwargs):
        if self.mode == 'shap-lime-top2':
            return self.shap_lime_top_n(2, **kwargs)
        elif self.mode == 'shap-lime-top3':
            return self.shap_lime_top_n(3, **kwargs)            
        else:
            return self.endorse_custom(**kwargs)

    ############ Endorsement batch by batch ############
    # Customize your methods here
    #

    def shap_lime_top_n(self,topn, **kwargs):
        idx, x,y0,y = kwargs['data_batch']
        y_pred = torch.argmax(y,dim=1)

        out = {}
        b = x.shape[0] # batch size
        for i in range(b):
            # Make sure to feed input one by one (not by batches)
            x_ = x[i:i+1] # but keep dim, so the shape is (1,*) 
            idx_ = idx[i].item()

            attr_kshap, attr_idx_kshap = self.kshap(x_, y_pred[i], topn=topn)
            attr_lime, attr_idx_lime = self.lime(x_, y_pred[i], topn=topn) 
            endorsement = self.endorse(*[attr_idx_kshap, attr_idx_lime])
            isCorrect = y_pred[i].item() == y0[i].item()

            if idx_ in out: raise Exception('duplicate index?!')

            out.update({ idx_ : {'endorsement': endorsement, 'isCorrect': isCorrect, 'y0': y0[i].item()} })
            
            if 'DEV_ITER' in self.kwargs:
                if self.kwargs['DEV_ITER'] > 0:
                    if i>=4: break
        return out

    def endorse_custom(self, *args, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)


class ClassifierEndorsementVisualizer():
    def __init__(self, ):
        super(ClassifierEndorsementVisualizer, self).__init__()
            
    @staticmethod
    def build_graph_nodes_and_edges(features, kfold_results, NODE_SPLIT_FILE, EDGE_SPLIT_FILE, correct_only=False):
        node_id = 0
        df_ = {'id':[], 'Label':[], 'y0':[], 'level':[]}
        df_edge_ = {'Source':[], 'Target':[], 'Weight':[], 'y0':[]}
        map_to_id = {
            # (feature_name,y0,level): node_id
        }

        ########### ADDING NODES ###########
        for idx_, ends in kfold_results.items():
            # example: 
            # ends = { 
            #  'endorsement': defaultdict(<class 'int'>, {8: 2, 10: 2, 9: 1, 15: 1}), 
            #  'isCorrect': False, 
            #  'y0': 0 }

            if correct_only:
                if not int(ends['isCorrect']): continue

            y0 = ends['y0']                
            for feature_idx, level in ends['endorsement'].items():
                ADD_NODE = True
                feature_name = features[feature_idx]

                if (feature_name,y0,level) in map_to_id: 
                    ADD_NODE = False

                if ADD_NODE:
                    map_to_id[(feature_name,y0,level)] = node_id

                    df_['id'].append(node_id)
                    df_['Label'].append(feature_name)
                    df_['y0'].append(y0)
                    df_['level'].append(level)
                    node_id += 1

        ########### ADDING EDGES ###########
        for idx_, ends in kfold_results.items():
            if correct_only:
                if not int(ends['isCorrect']): continue

            y0 = ends['y0']
            # we need to link all the nodes to each other: every pair of nodes one edge.
            # Note that the graph must be UNDIRECTED:
            #   look at how we loop thru index below, it's gonna create meaningless bias if directed.
            e_to_list = [(feature_idx, level) for feature_idx, level in ends['endorsement'].items() ]
             
            n_ef = len(e_to_list)
            for igx in range(n_ef):
                for igy in range(igx+1,n_ef):
                    feature_idx, level = e_to_list[igx]
                    feature_name = features[feature_idx]
                    df_edge_['Source'].append(map_to_id[(feature_name, y0, level)] )
                    
                    feature_idx, level = e_to_list[igy]
                    feature_name = features[feature_idx]
                    df_edge_['Target'].append(map_to_id[(feature_name, y0, level)])
                    
                    df_edge_['Weight'].append(1.0)
                    df_edge_['y0'].append(y0)

        pd.DataFrame(df_).to_csv(NODE_SPLIT_FILE, index=False)
        pd.DataFrame(df_edge_).to_csv(EDGE_SPLIT_FILE, index=False)   

    @staticmethod
    def build_endorsement_feature_view(features, kfold_results, FEATURE_VIEW_SUFFIX, correct_only=False, print_save_dir=False):
        n_features = len(features)
        data = {}

        #######################################################
        # This part only counts the max vote for better display
        # To-do. It looks redundant, edit it away in later versions
        max_vote = 0
        for idx_, ends in kfold_results.items():
            # print(idx_, ends)
            # ends = { 
            #  'endorsement': {8: 2, 10: 2, 9: 1, 15: 1}, 
            #  'isCorrect': False, 
            #  'y0': 0 }
            if correct_only:
                if not int(ends['isCorrect']): continue            
            for feature_idx, v in ends['endorsement'].items():
                if v> max_vote: max_vote = v
        #######################################################
        
        all_y0 = []
        for idx_, ends in kfold_results.items():
            if correct_only:
                if not int(ends['isCorrect']): continue

            y0 = ends['y0']
            if y0 not in data:
                all_y0.append(y0)
                data[y0] = {v : [0 for _ in range(n_features)] for v in range(1,max_vote+1)}
            for feature_idx, v in ends['endorsement'].items():  
                data[y0][v][feature_idx] += 1


        ########################################################
        # To-do. This part looks unnecessary.
        summary = {y0:{ feature_idx:0 for feature_idx in range(len(features))} for y0 in all_y0}
        for y0 in all_y0:
            for feature_idx in range(len(features)):
                for v in data[y0]:
                    summary[y0][feature_idx] += data[y0][v][feature_idx]
        """ Summary is like this:
        Total number of votes for each feature 0,1,...,n_features=6 for three classes [0,1,2]
        {
          1: {0: 229, 1: 213, 2: 178, 3: 162, 4: 107, 5: 71, 6: 0}, 
          0: {0: 432, 1: 408, 2: 357, 3: 309, 4: 231, 5: 131, 6: 0}, 
          2: {0: 210, 1: 201, 2: 188, 3: 141, 4: 111, 5: 61, 6: 0}
        }    
        """     
        ########################################################   

        import textwrap, re
        def fwrap(x):
            x = textwrap.fill(x.get_text(), 21)
            if len(x)>37:
                x = x[:37] +'...'
            return x 
        feature_display_names = [re.sub('_',' ',f) for f in features]

        for y0 in all_y0:
            df = pd.DataFrame(data[y0], index=feature_display_names)

            plt.rc('font', **{'size':12})
            df.plot(kind='barh', stacked=True, alpha=0.47, figsize=(7,int(0.7*n_features)))
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.gca().set_xlabel("total #endorsements")
            plt.gca().set_yticklabels(map(fwrap, plt.gca().get_yticklabels()))
            plt.legend(prop={'size':11}, framealpha=0.27, title='endorsement level', ) 
            plt.tight_layout()

            SAVE_DIR = FEATURE_VIEW_SUFFIX + f'-fv-gt{str(y0)}.png' 
            if print_save_dir: print(SAVE_DIR)
            plt.savefig(SAVE_DIR)   
            plt.close()

################################################
#          Classifier + Endorsement
################################################

class StandardXAIeClassifier(StandardAssemblyLineClassifier, StandardEndorsement):
    def __init__(self, DIRS, **kwargs):
        super(StandardXAIeClassifier, self).__init__(DIRS, **kwargs)
        self.mode = self.config['endorsement_mode']

    def endorse_selected_models(self, model_selection='auto', **kwargs):
        verbose = kwargs['verbose'] if 'verbose' in kwargs else 100
        DEV_ITER = kwargs['DEV_ITER'] if ' DEV_ITER' in kwargs else 0

        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        # print(branches, metric_types_per_branch) # like ['main-branch'] [['acc', 'f1']]

        # there's only 1 thing in the following: branch='main'
        for i,branch in enumerate(branches): 
            metric_types = metric_types_per_branch[i]

            # we use reduced dataset because we assume our training dataset is very large
            results_ = self.endorse_branch_reduced_train_dataset(branch, metric_types, 
                verbose=verbose, DEV_ITER=DEV_ITER )
            
            results = self.endorse_branch(branch, metric_types, 
                verbose=verbose, DEV_ITER=DEV_ITER )  
            
            # merge both of the above!
            for m in metric_types:
                results['best.'+m]['train'] = results_['best.'+m]['train']

            ENDORSE_FILE_NAME = 'endorse.branch-main.data'  
            joblib.dump(results, os.path.join(self.DIRS['ENDORSE_RESULT_DIR'],ENDORSE_FILE_NAME))

    def get_dataloader(self, *args, **kwargs):
        # implement downstream, depends on data
        # see for example setup_interface1.py, StandardClassifier()
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)        
        
    def endorse_branch(self, branch, metric_types, verbose=0, DEV_ITER=0):
        print('endorse_branch...')
        splits = ['val','test']
        """
        In this standard pipeline, a very large training dataset is validated and tested  
        against val/test datasets that contain only a small fraction (but sufficient) 
        number of data points. We compute endorsements only on val/test sets for brevity
        """

        results = {}
        for m in metric_types:
            results['best.'+str(m)] = {}
            for split in splits:
                results['best.'+str(m)][split] = {} 

        bvt = self.load_best_models()

        for m in metric_types:
            for split in splits:
                model = bvt.get_best_model(m)
                loader = self.get_dataloader(split=split, shuffle=False)
                results['best.'+str(m)][split] = self.endorse_batchwise(model, loader, 
                    endorsement_mode=self.config['endorsement_mode'], 
                    description= f"""<endorse> {branch} | data:{'%-5s'%(str(split))} | {"best.%-10s"%(str(m))}""",
                    DEV_ITER=DEV_ITER,
                    verbose=verbose
                )
        return results

    def endorse_branch_reduced_train_dataset(self, branch, metric_types, verbose=0, DEV_ITER=0):
        print('endorse_branch_reduced_train_dataset...')
        split = 'train'
        """
        In this standard pipeline, we reduce the large training dataset and use them for 
        our endorsement process. The structure of this function is similar to endorse_branch()
        """

        results = {}
        for m in metric_types:
            results['best.'+str(m)] = {}
            results['best.'+str(m)][split] = {} 

        bvt = self.load_best_models()

        for m in metric_types:
            model = bvt.get_best_model(m)
            
            # instead of using a normal get_dataloader(), we use the reduced version
            # loader = self.get_dataloader(split=split, shuffle=False)
            loader = self.get_dataloader_reduced_trainset(shuffle=False) # implemented downstream. See setup_interface1.py, for example
            
            results['best.'+str(m)][split] = self.endorse_batchwise(model, loader, 
                endorsement_mode=self.config['endorsement_mode'], 
                description= f"""<endorse> {branch} | data:{'%-5s'%(str(split))} | {"best.%-10s"%(str(m))}""",
                DEV_ITER=DEV_ITER,
                verbose=verbose
            )
        return results
        raise Exception('bibi')

           

# ******* Visualization ******** #
class StandardClassifierEndorsementVis(StandardAssemblyLineClassifier, ClassifierEndorsementVisualizer):
    def __init__(self, DIRS, **kwargs):
        self.data_cache = None
        if 'DATA_CACHE_DIR' in DIRS:
            self.data_cache = joblib.load(DIRS['DATA_CACHE_DIR'])
        else:
            print("No data cache is set (you're probably in some testing mode).")
        super(StandardClassifierEndorsementVis, self).__init__(DIRS, **kwargs)

    def set_config(self, **kwargs): pass
    def set_dataset(self, **kwargs): pass  
    def set_model(self, **kwargs): pass  
    def set_components(self, **kwargs): pass    

    @printfunc
    def visualize_endorsement_selected_models(self, model_selection='auto', feature_mode=None, **kwargs):
        print('visualize_endorsement...')

        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        for i,branch in enumerate(branches):
            ENDORSE_RESULT_DIR = os.path.join(self.DIRS['ENDORSE_RESULT_DIR'], 'endorse.branch-main.data')
            result_file = glob.glob(ENDORSE_RESULT_DIR)[0]

            results = joblib.load(result_file)
            for m in metric_types_per_branch[i]:
                model_type = 'best.'+str(m)
                self.visualize_endorsement_by_model(branch, model_type, results[model_type], feature_mode=feature_mode, correct_only=False)
                self.visualize_endorsement_by_model(branch, model_type, results[model_type], feature_mode=feature_mode, correct_only=True)
        return 'done visualizing endorsement...'

    def visualize_endorsement_by_model(self, branch, model_type, results_by_model, feature_mode=None, correct_only=False):
        splits = ['train','val','test'] # see endorse_branch(self,...)

        if feature_mode is None:
            features = self.data_cache['features']
        elif feature_mode == 'Token+Num': 
            TOKEN_FEATURES = self.data_cache['TOKEN_FEATURES']
            NUMERICAL_FEATURES = self.data_cache['NUMERICAL_FEATURES']
            features = list(TOKEN_FEATURES) + list(NUMERICAL_FEATURES) # yes, in this order        

        for split in splits:
            nametag = split
            if correct_only: nametag += '-correct-only'
            ENDORSE_VIS_SPLIT_DIR = os.path.join(self.DIRS['ENDORSE_VIS_DIR'], split)
            os.makedirs(ENDORSE_VIS_SPLIT_DIR, exist_ok=True)  
            self.build_graph_nodes_and_edges(features, results_by_model[split], 
                os.path.join(ENDORSE_VIS_SPLIT_DIR, f'{branch}.{model_type}.{nametag}-nodes.csv'), 
                os.path.join(ENDORSE_VIS_SPLIT_DIR, f'{branch}.{model_type}.{nametag}-edges.csv'), correct_only=correct_only)

            FEATURE_VIEW_SUFFIX = os.path.join(ENDORSE_VIS_SPLIT_DIR, f'{branch}.{model_type}.{nametag}')
            self.build_endorsement_feature_view(features, results_by_model[split], FEATURE_VIEW_SUFFIX)

        ##########################################################################
        # Repeat the above, but now we join all data together.
        ##########################################################################
        results_joined = {}
        for split in splits:
            results_joined.update(results_by_model[split])
        
        nametag = 'all'
        if correct_only: nametag += '-correct-only'
        ENDORSE_VIS_ALL_DIR = os.path.join(self.DIRS['ENDORSE_VIS_DIR'],'all')
        os.makedirs(ENDORSE_VIS_ALL_DIR, exist_ok=True)  
        self.build_graph_nodes_and_edges(features, results_joined, 
            os.path.join(ENDORSE_VIS_ALL_DIR , f'{branch}.{model_type}.{nametag}-nodes.csv'), 
            os.path.join(ENDORSE_VIS_ALL_DIR , f'{branch}.{model_type}.{nametag}-edges.csv'),
            correct_only=correct_only)  

        FEATURE_VIEW_SUFFIX = os.path.join(ENDORSE_VIS_ALL_DIR, f'{branch}.{model_type}.{nametag}')
        self.build_endorsement_feature_view(features, results_by_model[split], FEATURE_VIEW_SUFFIX)        

################################################
#       kFold Classifier + Endorsement
################################################

class kFoldXAIeClassifier(kFoldAssemblyLineClassifier, StandardEndorsement ):
    """
    Extension to kFoldAssemblyLineClassifier specifically designed to perform XAI endorsement
    """
    def __init__(self, DIRS, **kwargs):
        super(kFoldXAIeClassifier, self).__init__(DIRS, **kwargs)
        self.mode = self.config['endorsement_mode']

    def endorse_selected_models(self, model_selection='auto', **kwargs):
        assert(self.number_of_folds>=0) # dataset needs to be initiated first
        kfold_list, metric_types_per_kfold = self.select_models(model_selection, **kwargs)

        for i,k in enumerate(kfold_list):
            kfold_results = self.endorse_kth_fold(k, metric_types_per_kfold[i], 
                verbose=kwargs['verbose'], DEV_ITER=kwargs['DEV_ITER'])  

            ENDORSE_FILE_NAME = self.get_result_name_by_k(k, 'endorse')  
            joblib.dump(kfold_results, os.path.join(self.DIRS['ENDORSE_RESULT_DIR'],ENDORSE_FILE_NAME))

    def get_dataloader(self, *args, **kwargs):
        # implement downstream, depends on data
        # see for example setup_interface1.py, kFoldClassifier()
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def endorse_kth_fold(self, k, metric_types, verbose=0, DEV_ITER=0):
        splits = ['train','val','test']

        kfold_results = {}
        for m in metric_types:
            kfold_results['best.'+str(m)] = {}
            for split in splits:
                kfold_results['best.'+str(m)][split] = {} 

        bvt = self.load_kfold_best_models(k)

        for m in metric_types:
            for split in splits:
                model = bvt.get_best_model(m)
                loader = self.get_dataloader(k, split=split, shuffle=False)
                kfold_results['best.'+str(m)][split] = self.endorse_batchwise(model, loader, 
                    endorsement_mode=self.config['endorsement_mode'], 
                    description= f"""<endorse> k:{k} | data:{'%-5s'%(str(split))} | {"best.%-10s"%(str(m))}""",
                    DEV_ITER=DEV_ITER,
                    verbose=verbose
                )
        return kfold_results

# ******* Visualization ******** #
class kFoldClassifierEndorsementVis(kFoldAssemblyLineClassifier, ClassifierEndorsementVisualizer):
    def __init__(self, DIRS, **kwargs):
        self.data_cache = None
        if 'DATA_CACHE_DIR' in DIRS:
            self.data_cache = joblib.load(DIRS['DATA_CACHE_DIR'])
        else:
            print("No data cache is set (you're probably in some testing mode).")
        super(kFoldClassifierEndorsementVis, self).__init__(DIRS, **kwargs)

    def set_config(self, **kwargs): pass
    def set_dataset(self, **kwargs): pass
    def set_model(self, **kwargs): pass
    def set_components(self, **kwargs): pass

    @printfunc
    def visualize_endorsement_selected_models(self, model_selection='auto', feature_mode=None, **kwargs):
        # kfold_list: like [0,1,2,3,4]
        print('visualize_endorsement...')

        kfold_list, metric_types_per_kfold = self.select_models(model_selection, **kwargs)
        for i,k in enumerate(kfold_list):
            ENDORSE_RESULT_DIR = os.path.join(self.DIRS['ENDORSE_RESULT_DIR'], self.get_result_name_by_k(k, 'endorse'))
            result_file = glob.glob(ENDORSE_RESULT_DIR)[0]

            kfold_results = joblib.load(result_file)
            for m in metric_types_per_kfold[i]:
                model_type = 'best.'+str(m)
                self.visualize_endorsement_by_model(k, model_type, kfold_results[model_type], feature_mode=feature_mode, correct_only=False)
                self.visualize_endorsement_by_model(k, model_type, kfold_results[model_type], feature_mode=feature_mode, correct_only=True)
        return 'done visualizing endorsement...'

    def visualize_endorsement_by_model(self, k, model_type, kfold_results_by_model, feature_mode=None, correct_only=False):
        splits = ['train','val','test']

        if feature_mode is None:
            features = self.data_cache['features']
        elif feature_mode == 'Token+Num': 
            TOKEN_FEATURES = self.data_cache['TOKEN_FEATURES']
            NUMERICAL_FEATURES = self.data_cache['NUMERICAL_FEATURES']
            features = list(TOKEN_FEATURES) + list(NUMERICAL_FEATURES) # yes, in this order        

        for split in splits:
            nametag = split
            if correct_only: nametag += '-correct-only'
            ENDORSE_VIS_SPLIT_DIR = os.path.join(self.DIRS['ENDORSE_VIS_DIR'], split)
            os.makedirs(ENDORSE_VIS_SPLIT_DIR, exist_ok=True)  
            self.build_graph_nodes_and_edges(features, kfold_results_by_model[split], 
                os.path.join(ENDORSE_VIS_SPLIT_DIR, f'k-{k}.{model_type}.{nametag}-nodes.csv'), 
                os.path.join(ENDORSE_VIS_SPLIT_DIR, f'k-{k}.{model_type}.{nametag}-edges.csv'), correct_only=correct_only)

            FEATURE_VIEW_SUFFIX = os.path.join(ENDORSE_VIS_SPLIT_DIR, f'k-{k}.{model_type}.{nametag}')
            self.build_endorsement_feature_view(features, kfold_results_by_model[split], FEATURE_VIEW_SUFFIX)


        ##########################################################################
        # Repeat the above, but now we join all data together.
        ##########################################################################
        kfold_results_joined = {}
        for split in splits:
            kfold_results_joined.update(kfold_results_by_model[split])
        
        nametag = 'all'
        if correct_only: nametag += '-correct-only'
        ENDORSE_VIS_ALL_DIR = os.path.join(self.DIRS['ENDORSE_VIS_DIR'],'all')
        os.makedirs(ENDORSE_VIS_ALL_DIR, exist_ok=True)  
        self.build_graph_nodes_and_edges(features, kfold_results_joined, 
            os.path.join(ENDORSE_VIS_ALL_DIR , f'k-{k}.{model_type}.{nametag}-nodes.csv'), 
            os.path.join(ENDORSE_VIS_ALL_DIR , f'k-{k}.{model_type}.{nametag}-edges.csv'),
            correct_only=correct_only)  

        FEATURE_VIEW_SUFFIX = os.path.join(ENDORSE_VIS_ALL_DIR, f'k-{k}.{model_type}.{nametag}')
        self.build_endorsement_feature_view(features, kfold_results_by_model[split], FEATURE_VIEW_SUFFIX)

################################################
#        EEC: Extract Endorsement Core
################################################
"""
Here, the goal is to extract a subset of training samples from the training set,
and then test their results on the same model.
The subset is chosen based on the partitions created by endorsement. Example of partition:
all data points whose endorsements are {0:2,5:2,17:2} will be categorized into one partitition.
In other words, we group together all trainig data points whose prediction depend on features 0,5 and 17. 
Note: we will partition the data points further based on the ground-truth labels.
"""

class EEC():
    def __init__(self, ):
        super(EEC, self).__init__()
        # DC: data core
        # 'fraction' refers to the size of subset compared to total training dataset
        # We will 'increment' this fraction and test the training performance based on our data subsets
        self.DC = {
            'start': 0.2,
            'increment': 0.1,
            'max': 0.901
        }

    ################### Partition #########################
    # Before extracting cores, we need to partition training
    # data based on the endorsements

    @printfunc
    def eec_partition_selected_models(self, model_selection='auto', feature_mode=None, prefix='branch', **kwargs):
        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        for i,branch in enumerate(branches):
            ENDORSE_RESULT_DIR = os.path.join(self.DIRS['ENDORSE_RESULT_DIR'], f'endorse.{prefix}-{branch}.data')
            
            result_file = glob.glob(ENDORSE_RESULT_DIR)[0]
            results = joblib.load(result_file)              

            for m in metric_types_per_branch[i]:
                model_type = 'best.'+str(m)
                eec_recipe = self.eec_partition_by_model(branch, model_type, results[model_type], prefix=prefix, feature_mode=feature_mode)

        return 'done partitioning selected model...'

    def eec_partition_by_model(self,branch, model_type, results_by_model, prefix='branch',  feature_mode=None):
        # we are interested in only the 'train' split now as we are trying to extract training data subset
        split = 'train'

        EEC_PARTITION_SUFFIX = os.path.join(self.DIRS['EEC_RESULT_DIR'], f'{prefix}-{branch}.{model_type}.partition')
        EEC_VISUALIZATION_SUFFIX = os.path.join(self.DIRS['EEC_VIS_DIR'], f'{prefix}-{branch}.{model_type}.partition')

        # partitions: a dictionary with key <<frozen set of endorsement + y0>>
        # and value <<idx_ raw indices of data points>>
        partitions = self.eec_build_partition(results_by_model[split], feature_mode=feature_mode)        
        summary = self.eec_summarize_partitions(partitions, EEC_VISUALIZATION_SUFFIX + '.summary')

        """ !! save.result !! """        
        eec_recipe = {
            'partitions':partitions,
            'partitions_summary': summary,
        }
        joblib.dump(eec_recipe, EEC_PARTITION_SUFFIX + '.eecr')
        return eec_recipe

    def eec_build_partition(self, results, feature_mode=None):
        if feature_mode is None:
            features = self.data_cache['features']
        elif feature_mode == 'Token+Num': 
            TOKEN_FEATURES = self.data_cache['TOKEN_FEATURES']
            NUMERICAL_FEATURES = self.data_cache['NUMERICAL_FEATURES']
            features = list(TOKEN_FEATURES) + list(NUMERICAL_FEATURES) # yes, in this order 

        partitions = {}
        for idx_, vals in results.items():    
            # recall: idx_ is raw index straight out of the csv!

            if int(vals['isCorrect'])==0: continue # exclude all wrong predictions

            # idx_: the index of raw data
            endorsement = vals['endorsement'] # like defaultdict(<class 'int'>, {12: 2, 16: 2, 3: 1, 5: 1})
            y0 = vals['y0'] # ground-truth value 

            partition_group_key = []
            for feature_idx, n_endorse in endorsement.items():
                partition_group_key.append((feature_idx, n_endorse))
            partition_group_key.append(('y0', y0))

            # let's convert it into a frozenset. Why set and frozen? Because we want uniqueness.
            # With unique entry, {17:2, 5:1} is the same as {5:1, 17:2}, and we will not screw up
            # the semantics of our endorsement. It needs to be frozen because we are storing them
            # as dictionary keys.
            partition_group_key = frozenset(partition_group_key)

            if not partition_group_key in partitions:
                partitions[partition_group_key] = []

            partitions[partition_group_key].append(idx_)
        return partitions

    @staticmethod
    def compute_simple_descriptive_stats(partition_sizes):
        return {
            'mean': np.mean(partition_sizes),
            'median': np.median(partition_sizes),
            'q0.75': np.quantile(partition_sizes, 0.75),
        }  

    def eec_summarize_partitions(self, partitions, SUMMARY_DIR):
        total_training_size = 0
        partition_sizes = []
        
        txt = open(SUMMARY_DIR + '.txt', 'w')
        txt.write('%s <- %s\n'%(str('partition size'),str('partition unique key')))
        for partition_group_key, partition in partitions.items(): 
            partition_size = len(partition)

            total_training_size +=  partition_size
            partition_sizes.append(partition_size)

            partition_size_and_key = str('%-5s <- '%(str(partition_size))) + str(set(partition_group_key)) + '\n'
            txt.write(partition_size_and_key)

        summary = {}
        summary['partition_sizes'] = np.sort(partition_sizes)
        summary['partition_descriptive_stats'] = self.compute_simple_descriptive_stats(partition_sizes)
        summary['quantiles'] = []            

        """ ####### CORE SEARCH AND COUNTING #######
        !!Caveat!! "zero quantile problem".
        This counting mechanism is known to yield ZERO quantiles in a few cases. 
        From what we understand, if the partition_sizes are imbalancely distributed,
        the only possible quantiles are those that either:
        1. produce too few cores 
        2. produce too many cores (more than self.DC['max']) so that the reduction in the number of training set is just too small to matter.
        In these cases, they are excluded. Sometimes, we see a situation where ALL cases are excluded!


        ******* How to spot zero quantile problem? *******
        See eec.result folder. For each XXX-YYY.best.<<metric_type>>.partition.eecr,
        you should see some of these folders:
                 XXX-YYY.best.<<metric_type>>.partition.type-ZZZ
        If these folders with the given <<metric_type>> are missing, it means the model obtained from your best <<metric_type>> (perhaps acc, recall etc) have the problems

        """        

        running_dc_fraction = 0. + self.DC['start']
        txt.write('\n(q-th,quantile)  -> n core data [[fraction]]\n')
        qs, quantiles, core_sizes, core_fractions = [], [], [], []
        for q in np.linspace(0.5, 0.9, 21):
            quantile = np.round(np.quantile(partition_sizes, q))
            if quantile in quantiles: continue # we don't want duplicates

            core_size = np.sum(np.clip([x for x in partition_sizes], 1, quantile)).astype(int)
            core_fraction = np.round(core_size/total_training_size,3)
            
            qs.append(q)
            quantiles.append(quantile)
            core_sizes.append(core_size)
            core_fractions.append(core_fraction)            

            if core_fraction >= running_dc_fraction:
                if core_fraction > self.DC['max']: break

                conf = '%-14s'%(f'({np.round(q,3)}, {quantile})')
                subset_info_by_q = f' {conf} ->  {core_size} [[{core_fraction}]]\n'
                
                # quantile will be used for the actual EEC process
                # core_fraction, core_size will be used for double checking 
                summary['quantiles'].append({
                    'q':q,  'quantile': quantile, 
                    'core_fraction': core_fraction, 'core_size': core_size
                    })

                txt.write(subset_info_by_q)
                running_dc_fraction += self.DC['increment']


        txt.write(f'\ntotal no. of data : {total_training_size}\n')
        txt.write(f"\nsummary descriptive stats : {json.dumps( summary['partition_descriptive_stats'])}\n")
        txt.close()

        self.plot_and_save_partition_summary(partition_sizes, SUMMARY_DIR)
        self.plot_quantiles(qs, quantiles, core_fractions, core_sizes, SUMMARY_DIR)

        return summary

    # ###################### EEC ##############################
    # Peform the actual Extraction of Endorsement Cores here.
    # For each partition, we find a subset of data that will 
    # serve as the cores for that partition.
    # The new subset of data are hoped to be more efficient
    # for training compared to the entire data. 


    @staticmethod
    def verify_same_label_within_partition(y0, THIS_PARTITION_LABEL):
        if THIS_PARTITION_LABEL is None:
            THIS_PARTITION_LABEL = y0
        else:
            assert(THIS_PARTITION_LABEL==y0)

    def eec_selected_models(self, model_selection='auto', prefix='branch', **kwargs):
        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        for i,branch in enumerate(branches):
            for m in metric_types_per_branch[i]:
                model_type = 'best.'+str(m)            
                self.eec_by_model(branch, model_type, prefix=prefix, **kwargs)

    def eec_by_model(self, branch, model_type, prefix='branch', **kwargs):
        pname = f'{prefix}-{branch}.{model_type}.partition'
        EEC_PARTITION_SUFFIX = os.path.join(self.DIRS['EEC_RESULT_DIR'], pname)        

        eec_recipe = joblib.load(EEC_PARTITION_SUFFIX + '.eecr')

        partitions = eec_recipe['partitions']
        summary = eec_recipe['partitions_summary']
        """ summary is like
        {  
            'partition_sizes': array([ 1,  1,  1, ..., 31, 66, 68, 69, 70, 89]), 'partition_descriptive_stats': {'mean': 13.688888888888888, 'median': 3.0, 'q0.75': 12.0}, 
            'quantiles': [
                {'q': 0.52, 'quantile': 3.9, 'core_fraction': 0.201, 'core_size': 124}, 
                {'q': 0.72, 'quantile': 9.0, 'core_fraction': 0.326, 'core_size': 201}, 
                ...
            ]}
        """        
        for i,eec_param in enumerate(summary['quantiles']):
            self.build_endorsement_core_data_subset(branch, model_type, eec_param, partitions, EEC_PARTITION_SUFFIX, **kwargs)

    def build_endorsement_core_data_subset(self, *args, **kwargs):
        """ !! Need to implement downstream since it depends on the data

        Prototypical example: see kFoldClassifier object in setup_interface1.py 

        ####### Basic structure #######
        df <-- initiate some empty dataframe here
        for pkey, indices in partitions.items():    
            df_op = self.eec_partition_to_eec_data(**kwargs)        
            df <-- append df_op to df
        df.to_csv(EEC_DATA_DIR, index=False)
        
        """
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def eec_partition_to_eec_data(self, *args, **kwargs):
        # !! Need to implement downstream since it depends on the data
        # A function to process each partition
        # Is a component of build_endorsement_core_data_subset() 
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    @printfunc
    def post_eec_train_val_test(self, model_selection='auto', prefix='branch',**kwargs):
        """
        // ************* FINAL STEP, yay! ************* //
        We've extracted subset of data based on partitions via EEC (Extract Endorseemnt Core).
        Now we want to train using these reduced data and hope that the smaller core subset data we have extracted are good enough.
        """
        branches, metric_types_per_branch = self.select_models(model_selection, **kwargs)
        for i,branch in enumerate(branches):
            for m in metric_types_per_branch[i]:
                model_type = 'best.'+str(m)          
                self.post_eec_train_val_test_per_branch_per_model(branch, model_type, prefix=prefix, **kwargs)
        return 'done'

    def post_eec_train_val_test_per_branch_per_model(self, branch, model_type, prefix='branch', **kwargs):
        pname = f'{prefix}-{branch}.{model_type}.partition'
        EEC_PARTITION_SUFFIX = os.path.join(self.DIRS['EEC_RESULT_DIR'], pname)        

        eec_recipe = joblib.load(EEC_PARTITION_SUFFIX + '.eecr')
        summary = eec_recipe['partitions_summary']

        for i,eec_param in enumerate(summary['quantiles']):
            t = threshold = int(eec_param['quantile'])

            # See notes about implementations of post_eec_train_val and post_eec_test below
            for eec_sub_type in self.config['eec_modes']:
                self.post_eec_train_val(branch, model_type, t, EEC_PARTITION_SUFFIX, eec_sub_type, prefix=prefix, **kwargs)
                self.post_eec_test(branch,model_type, t, EEC_PARTITION_SUFFIX, eec_sub_type, prefix=prefix, **kwargs)

    """ ======= Note =======
    !! Need to implement downstream, the pipeline depends on data and the previous train/val/test pipeline.
    def post_eec_train_val(self, *args, **kwargs):    
        #  Training, val and test process should be the same as the train_val_test() pipeline in factory.py except for minor logistical differences. For example, in our implementation of endorsement

    def post_eec_test(self, *args, **kwargs):
        # Yup, after train and val, do the test. Implement the details downstream
        # After this, we're left with visualization of all the final results.
    """

class EECPartitionVisualizer():
    @staticmethod
    def plot_quantiles(qs, quantiles, core_fractions, core_sizes, SUMMARY_DIR):
        font = {'size': 12}
        plt.rc('font', **font)
        plt.figure(figsize=(6,3))
        ax1 = plt.gcf().add_subplot(1,2,1)
        plt.gca().plot(qs, core_fractions, c='k',linewidth=0.5)
        plt.gca().set_xlabel('q-th')
        plt.gca().set_ylabel('data fraction')

        ax2 = ax1.twinx()
        thisc = (0.97,0.57,0.77)
        plt.gca().plot(qs, quantiles, c=thisc, alpha=0.27, linewidth=2, label="quantile")
        plt.gca().tick_params(axis='y', colors=thisc, labelsize=9)
        plt.legend(fontsize=10)

        plt.gcf().add_subplot(1,2,2)
        plt.gca().plot(quantiles, core_sizes, alpha=0.77, label='data size')
        # plt.gca().tick_params(axis='y', colors=thisc, labelsize=9)
        plt.legend(fontsize=10)
        plt.gca().set_xlabel('quantile')

        plt.tight_layout()
        plt.savefig(SUMMARY_DIR + '.eec.png')
        plt.close()            

    @staticmethod
    def plot_and_save_partition_summary(partition_sizes, SUMMARY_DIR):
        font = {'size': 12}
        plt.rc('font', **font)

        bins = get_uniform_hist_bin(partition_sizes,binwidth=1)
        font = {'size': 21}
        plt.rc('font', **font)
        plt.figure(figsize=(12,10))
        def plot_(partition_sizes, yscale=None, ylabel='no. of partitions', figure_pos_offset=0):
            ax1 = plt.gcf().add_subplot(2,2,1+ figure_pos_offset)
            freq, outbins, patches = plt.gca().hist(np.array(partition_sizes), alpha=0.3, 
                color=(0.27,0.77,0.27), rwidth=1,bins=bins)

            plt.gca().set_xlabel('partition size\n(no. of idx inside a particular partition)')
            plt.gca().set_ylabel(ylabel)
            if yscale is not None:
                plt.gca().set_yscale(yscale)

            def set_fixed_ticks(bins, nticks=7):
                xticks = []
                spacing = np.ceil((bins[-1]-bins[0])/nticks)
                for i in range(1,1+nticks):
                    xticks.append(spacing*i)
                plt.gca().set_xticks(xticks)

            def annotate_colored_random_height(freq, outbins, patches):
                binwidth = outbins[1]-outbins[0]
                color_transition = {'green':'red','red':'blue','blue':'green'}  
                c = 'green'
                maxh = 0.1*np.max(freq)
                for i, (fr, x0, patch) in enumerate(zip(freq, outbins, patches)):
                    height = int(freq[i])
                    if height == 0: continue
                    if i%2>0: continue

                    c = color_transition[c]
                    ypos = height + (i%4) + maxh * np.random.uniform(0.,0.5)

                    ymin, ymax = 0, ypos
                    plt.vlines(x0+binwidth/2, ymin, ymax, colors=(0.47,0.47,0.47), linestyles='dashed', linewidth=0.5)

                    plt.annotate(f"{int(x0)}", xy=(x0 + binwidth/2., ypos), xytext=(0,0.2), color=c,
                        textcoords="offset points", ha = 'center', va = 'bottom', alpha=0.5, size=7)

            set_fixed_ticks(bins)
            annotate_colored_random_height(freq, outbins, patches)

            ax2 = ax1.twinx()
            thisc = (0.77,0.17,0.77)
            ax2.hist(np.array(partition_sizes), color=thisc, cumulative=True, histtype="step", alpha=0.17, rwidth=1,bins=bins, label='total no. of partitions')
            ax2.tick_params(axis='y', colors=thisc, labelsize=9)
            lgnd = plt.legend(prop={'size':9}, framealpha=0.27)
            for i in range(len(lgnd.legendHandles)):
                lgnd.legendHandles[i]._sizes = [11] 
                lgnd.legendHandles[i].set_alpha(0.17)

            plt.gcf().add_subplot(2,2,2+figure_pos_offset)
            plt.gca().hist(np.log10(partition_sizes), alpha=0.1, 
                color=(0.77,0.57,0.27), edgecolor=(0.77,0.27,0.27))
            plt.gca().set_xlabel('$log_{10}$(partition size)')
            if yscale is not None:
                plt.gca().set_yscale(yscale)
        
        plot_(partition_sizes, figure_pos_offset=0)
        plot_(partition_sizes, yscale='log', ylabel='$log_{10}$(no. of partitions)', figure_pos_offset=2)

        plt.tight_layout()
        plt.savefig(SUMMARY_DIR + '.png')
        plt.close()    

# ******* Standard EEC ******* #

class StandardXAIeClassifierEEC(StandardXAIeClassifier, EECPartitionVisualizer, EEC):
    def __init__(self, DIRS, **kwargs):
        EEC.__init__(self)
        super(StandardXAIeClassifierEEC, self).__init__(DIRS, **kwargs)

        self.data_cache = joblib.load(self.DIRS['DATA_CACHE_DIR'])

        self.DC['increment'] = 0.15  
        # bigger increment, less no. of post eec training. For faster completion, but results less fine.

# ******* kFold EEC ******* #

class kFoldXAIeClassifierEEC(kFoldXAIeClassifier, EECPartitionVisualizer, EEC):
    def __init__(self, DIRS, **kwargs):  
        EEC.__init__(self)                  
        super(kFoldXAIeClassifierEEC, self).__init__(DIRS, **kwargs)

        self.data_cache = joblib.load(self.DIRS['DATA_CACHE_DIR'])

