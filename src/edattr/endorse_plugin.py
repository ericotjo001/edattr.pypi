from .utils import *

from captum.attr import KernelShap, Lime
from collections import defaultdict
from edattr.endorse import edattr_to_feature_attr


class ComparisonEndorsement():
    """ This is an abstract class like StandardEndorsement in src/endorse.py
    We implement this adhoc for the sole purpose of method comparison.
        
    It is implemented as a plugin to other objects, and thus it will use 
    methods and properties from the main class of the objects (and everything 
    they inherit from other classes)

    !! Important !! Do not write any function that overrides any function in the 
    standard endorsement class objects.
    """
    def __init__(self):
        super(ComparisonEndorsement, self).__init__()

    def set_compare_result_dirs(self):
        metric_type = self.kwargs['best.metric']

        LABEL_DIR = self.DIRS['LABEL_DIR']
        PLUGIN_RESULT_DIR = os.path.join(LABEL_DIR, '_plugin.compare_result')
        os.makedirs(PLUGIN_RESULT_DIR, exist_ok=True)
        COMPARE_RESULT_DIR = os.path.join(PLUGIN_RESULT_DIR, f'compare_result-best.{metric_type}.json')
        COMPARE_RESULT_VIS_DIR = os.path.join(PLUGIN_RESULT_DIR, f'compare_result-best.{metric_type}.png')

        COMPARE_RESULT_ABSVAL_DIR = os.path.join(PLUGIN_RESULT_DIR, f'compare_result-best.{metric_type}.absval.json')
        COMPARE_RESULT_ABSVAL_VIS_DIR = os.path.join(PLUGIN_RESULT_DIR, f'compare_result-best.{metric_type}.absval.png')

        self.DIRS.update({
            'COMPARE_RESULT_DIR':COMPARE_RESULT_DIR,
            'COMPARE_RESULT_VIS_DIR':COMPARE_RESULT_VIS_DIR,   
            'COMPARE_RESULT_ABSVAL_DIR':COMPARE_RESULT_ABSVAL_DIR,
            'COMPARE_RESULT_ABSVAL_VIS_DIR':COMPARE_RESULT_ABSVAL_VIS_DIR,                        
            })

    def load_model_and_dataloader(self):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM)

    def compare_edattr_batchwise(self, absval=False):
        self.set_compare_result_dirs()

        if absval:
            COMPARE_RESULT_DIR = self.DIRS['COMPARE_RESULT_ABSVAL_DIR']
        else:
            COMPARE_RESULT_DIR = self.DIRS['COMPARE_RESULT_DIR']
        
        # This method is modelled after endorse_batchwise() in src/endorse.py
        m = self.kwargs['best.metric']
        model, loader = self.load_model_and_dataloader(m)

        from accelerate import Accelerator
        accelerator = Accelerator()
        model, loader, self.criterion = accelerator.prepare(model, loader, self.criterion)
        model.eval()        

        self.populate_attribution_models(model)
        progress = tqdm.tqdm(enumerate(loader), total=len(loader),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
            disable=False,)

        # e_batch = {}
        if self.mode in ['shap-lime-top2', 'shap-lime-top3']:
            comparison_results = {'agree': 0, 'total':0, 'kshap': {}, 'lime' : {}, 'edattr':{}}
        else:
            raise NotImplementedError('not implemented, indeed')
        
        for i,(idx, x,y0) in progress:
            progress.set_description('making comparison...')
            x = x.to(torch.float)
            y = model(x)

            self.endorse_batch_with_perturbation_comparison_(comparison_results, absval=absval, **{'data_batch':(idx, x,y0,y), 'model': model,})
            # e_batch.update(e_batch_)

        with open(COMPARE_RESULT_DIR,'w') as f:
            json.dump(comparison_results,f)
        print(f'results stored at {COMPARE_RESULT_DIR}')

    def endorse_batch_with_perturbation_comparison_(self, comparison_results, absval=False, **kwargs):
        # This method is modelled after endorse_batch_() in src/endorse.py

        if self.mode == 'shap-lime-top2':
            return self.shap_lime_top_n_with_PC(2, comparison_results, absval=absval, **kwargs)
        elif self.mode == 'shap-lime-top3':
            return self.shap_lime_top_n_with_PC(3, comparison_results, absval=absval, **kwargs)           
        else:
            return self.endorse_custom_with_PC(**kwargs)

    ############ Endorsement batch by batch ############
    # Customize your methods here, to match the methods in src/endorse.py
    #

    def shap_lime_top_n_with_PC(self, topn, comparison_results, absval=False, **kwargs):
        # PC: perturbation comparison
        idx, x,y0,y = kwargs['data_batch']
        y_pred = torch.argmax(y,dim=1)
        model = kwargs['model']
    
        b, n_features = x.shape
        for i in range(b):
            # Make sure to feed input one by one (not by batches)
            x_ = x[i:i+1] # but keep dim, so the shape is (1,*) 
            y_ = y[i:i+1]
            y0_ = y0[i:i+1]
            idx_ = idx[i:i+1].item()

            attr_kshap, attr_idx_kshap = self.kshap(x_, y_pred[i], topn=topn)
            attr_lime, attr_idx_lime = self.lime(x_, y_pred[i], topn=topn) 
            endorsement = self.endorse(*[attr_idx_kshap, attr_idx_lime])
            isCorrect = y_pred[i].item() == y0[i].item()
            
            ############## COMPARISON ############## 
            if set(attr_idx_kshap) == set(attr_idx_lime):
                comparison_results['agree'] += 1
            else:
                # print(endorsement) # like defaultdict(<class 'int'>, {0: 1, 1: 2, 3:1})
                edattr = edattr_to_feature_attr(endorsement, n_features)
                edattr = torch.tensor(edattr).unsqueeze(0) # just adjusting the format for consistency

                comparison_results['kshap'][idx_] = self.perturb_and_loss(x_, y_, y0_, model, attr_kshap, absval=absval)
                comparison_results['lime'][idx_] = self.perturb_and_loss(x_, y_, y0_, model, attr_lime, absval=absval)
                comparison_results['edattr'][idx_] = self.perturb_and_loss(x_, y_, y0_, model, edattr, absval=absval)
            comparison_results['total'] += 1
            ########################################

            if 'DEV_ITER' in self.kwargs:
                if self.kwargs['DEV_ITER'] > 0:
                    if i>=4: break

    def endorse_custom_with_PC(self, *args, **kwargs):
        raise NotImplementedError(UMSG_IMPLEMENT_DOWNSTREAM) 

    def perturb_and_loss(self, x_, y_, y0_, model, attr_, absval=False):        
        # print(x_.shape, y_.shape, attr_.shape) 
        # torch.Size([1, 4]) torch.Size([1, 3]) torch.Size([1, 4]) # ok
        # print('>>',x_) # recall: x_ is [TOKEN_FEATURES, NUMERICAL_FEATURES] in that order
        dict_leng = self.config['dict_leng']
        n_p = self.config['perturb_n']
        b = self.config['batch_size']
        assert(n_p>=b)
        # print('n_p >>', n_p) # ok
        n_token = len(self.TOKEN_FEATURES)
        n_num = len(self.NUMERICAL_FEATURES)
        # print('n_token >>', n_token) # ok

        # perturbation probability feature by feature
        # Max feature score has 0.75 chance of being perturbed after normalization. 
        # Any feature score below 0.1 will be set to have 0.1 chance of being perturbed. 
        prob = attr_.clone().squeeze().detach().cpu().numpy() 
        probnorm = np.max(np.abs(prob))
        if probnorm == 0: probnorm = 1.0
        prob = 0.75*prob/probnorm
        prob = np.clip(prob,0.1,None)

        delta_losses = [] # this is the values that we eventually use for comparison 
        # just wanna compute them batch by batch. No real reason, really, 
        # except large n_p may exceed memory
        n_ = int(np.ceil(n_p/b))
        for i in range(n_):
            start, end = i*b, (i+1)*b
            end = np.clip(end, None, n_p)

            this_b = end-start
            x_perturb = x_.clone().repeat((this_b,1))

            coin_flips = np.zeros(shape=(this_b,n_token+n_num)) # same size as x_perturb
            for col in range(n_token+n_num):
                coin_flips[:,col] = np.random.uniform(0,1,size=(this_b)) < prob[col]    
            coin_flips = torch.tensor(coin_flips).to(x_perturb.device)    
            # print(x_perturb.shape, coin_flips.shape) # they should be the same shape         
            # print(np.mean(coin_flips,axis=0),prob) # they are approximately equal 

            # With token_shifts, when the coin flips head for token features
            # the perturbation will be guaranteed. 
            token_shifts = range(1, dict_leng) 
            token_perturb = np.random.choice(token_shifts, size=(this_b,n_token))
            token_perturb = torch.tensor(token_perturb).to(x_perturb.device)
            x_perturb[:,:n_token] = (x_perturb[:,:n_token] + coin_flips[:,:n_token] * token_perturb)% dict_leng

            # With numerical array*num_shift, 
            # when the coin flips head, the numerical feature is set to either 0 or 
            # to the negative of its original value. 
            num_shift = [-1,-2]
            pmagnitude = np.random.choice(num_shift, size=(this_b,n_num))
            numerical_perturb = x_perturb[:,n_token:] * torch.tensor(pmagnitude).to(x_perturb.device)
            x_perturb[:,n_token:] = x_perturb[:,n_token:] + coin_flips[:,n_token:] * numerical_perturb

            yp_batch = model(x_perturb)
            # print(yp_batch.shape) # like (b,C) where C is # of classes

            y0_b = y0_.clone().repeat((this_b,))
            y_b = y_.clone().repeat((this_b,1))

            baseline_loss = self.criterion(y_b, y0_b).item()
            perturbed_loss = self.criterion(yp_batch, y0_b).item()

            delta_loss = perturbed_loss - baseline_loss
            # we expect good feature attribution methods to have higher delta loss.
            # Why? In the above, we perturb features with higher importance more often.
            # This means that, ideally, feature attribution methods who understand
            # which features are truly more important will suffer more disruption than
            # poorer methods.

            if absval: delta_loss = np.abs(delta_loss)

            delta_losses.append(delta_loss)

        avg_delta_loss = np.mean(delta_losses) # avg per single data point
        return avg_delta_loss

    def visualize_edattr_comparison(self, absval=False):
        self.set_compare_result_dirs()
        if not absval:
            COMPARE_RESULT_DIR = self.DIRS['COMPARE_RESULT_DIR']
            COMPARE_RESULT_VIS_DIR = self.DIRS['COMPARE_RESULT_VIS_DIR']
        else:
            COMPARE_RESULT_DIR = self.DIRS['COMPARE_RESULT_ABSVAL_DIR']
            COMPARE_RESULT_VIS_DIR = self.DIRS['COMPARE_RESULT_ABSVAL_VIS_DIR']

        with open(COMPARE_RESULT_DIR) as f:
            compare_results = json.load(f)
        pvalues = self.get_pvalues_package(compare_results, absval=absval)

        font = {'size': 16}
        plt.rc('font', **font)

        plt.figure(figsize=(2*len(compare_results)-3,7))
        plt.rcParams['legend.title_fontsize'] = 9
        counter = 0
        methods = []
        mean_marks = []
        for method_, result in compare_results.items():
            if method_ in  ['agree', 'total']: continue

            counter += 1
            methods.append(method_)

            # print(pvalues[method_].pvalue) # ok
            p_ = pvalues[method_].pvalue
            plotlabel = f'p={round(p_,4)}' if p_ is not None else p_ 

            avg_delta_losses = [ADL for _,ADL in result.items()]
            n_ = len(avg_delta_losses)
            x = np.zeros(shape=(n_,)) + counter + np.random.uniform(-0.1,0.1,size=(n_,)) 
            plt.gca().scatter(x,avg_delta_losses,label=plotlabel, alpha=0.27)

            mean_marks.append(np.mean(avg_delta_losses)) 
        plt.gca().set_xlim(0.5,counter+0.5)
        plt.gca().set_xticks([0.5]+list(range(1,counter+1)))
        plt.gca().set_xticklabels([""] + methods)

        plt.gca().plot(range(1,counter+1) , mean_marks, marker='^', c='r', linewidth=0.5)

        plt.gca().set_ylabel('Avg Delta Losses')
        offset_to_right = 0
        plt.legend(prop={'size': 9}, loc='center left', bbox_to_anchor=(1 + offset_to_right, 0.5), title="one way anova\nw.r.t. edattr")
        plt.tight_layout()
        plt.savefig(COMPARE_RESULT_VIS_DIR)
        print(f'Figure saved to {COMPARE_RESULT_VIS_DIR}')

    def get_pvalues_package(self, compare_results, absval=False):
        labelname = os.path.basename(self.DIRS['LABEL_DIR'])

        if absval:
            csv_filename = f'{self.kwargs["full_projectname"]}-compare-agg-pvalues.absval.csv'
        else:
            csv_filename = f'{self.kwargs["full_projectname"]}-compare-agg-pvalues.csv'

        PVALUE_AGG_CSV = os.path.join(self.DIRS['PROJECT_DIR'], csv_filename)

        if not os.path.exists(PVALUE_AGG_CSV):
            pvalue_agg = {'labelname':[]}
        else:
            pvalue_agg = pd.read_csv(PVALUE_AGG_CSV)
            pvalue_agg = pvalue_agg.to_dict()
            for x,y in pvalue_agg.items():
                pvalue_agg[x] = [y_ for _,y_ in y.items()] # just converting to list

        from scipy.stats import f_oneway
        control_group = [val for _,val in compare_results['edattr'].items()]

        class Obj(object):
            pvalue = None
            statistic = None
        dummy_ = Obj()
        pvalues = {'edattr': dummy_}
        
        for method_, result in compare_results.items():
            if method_ in  ['agree', 'total', 'edattr']: continue
            treatment_group = [val for _,val in compare_results[method_].items()]
            pvalues[method_] = f_oneway(treatment_group, control_group)
            # print(pvalues[method_]['pvalue']) # ok
            
            if not method_ in pvalue_agg:
                pvalue_agg[method_] = []

            pdisplay = None
            try:
                pdisplay = str(round(pvalues[method_].pvalue,4))
                if pvalues[method_].pvalue<0.05:
                    pdisplay += "*"    
            except:
                pdisplay = 'N/A'
            pvalue_agg[method_].append(pdisplay)

        pvalue_agg['labelname'].append(labelname)
        pvalue_agg = pd.DataFrame(pvalue_agg)
        pvalue_agg.to_csv(PVALUE_AGG_CSV, index=None)

        return pvalues


class ComparisonEndorsementClassifierK2(ComparisonEndorsement):
    """ We use this in setup_interface2.py
    Data Frame Types: TokenAndFloat DATAFRAME (see data.py)
    """
    def __init__(self, ):
        super(ComparisonEndorsementClassifierK2, self).__init__()

    def load_model_and_dataloader(self, metric_type):    
        split = self.kwargs['split']
        best_values_where = self.load_best_value_where() 
        # print(best_values_where) # {'acc': [0, 0.8541666666666666], 'f1': [0, 0.8125]}

        k = branch = best_values_where[metric_type][0]
        bvt = self.load_kfold_best_models(k)
        model = bvt.get_best_model(metric_type)
        loader = self.get_dataloader(k, split=split, shuffle=False)
        return model, loader

class ComparisonEndorsementClassifierS2(ComparisonEndorsementClassifierK2):
    """ We use this in setup_interface2.py
    Data Frame Types: TokenAndFloat DATAFRAME (see data.py)
    """
    def __init__(self, ):
        super(ComparisonEndorsementClassifierS2, self).__init__()

    def load_model_and_dataloader(self, metric_type):
        split = self.kwargs['split']
        best_values_where = self.load_best_value_where() 
             
        bvt = self.load_best_models()
        model = bvt.get_best_model(metric_type)
        loader = self.get_dataloader(split=split, shuffle=False)
        return model, loader

def aggregate_compare_results(DIRS, absval=False, **kwargs):
    print(f'aggregating comparison results! absval:{absval}'  )
    metric_type = kwargs['best.metric']

    y, labels = [], []
    aggname = kwargs['full_projectname']
    if absval:
        aggname = aggname + f'-compare-agg.{metric_type}.absval.png'
        resultname = f'compare_result-best.{metric_type}.absval.json'
    else:
        aggname = aggname + f'-compare-agg.{metric_type}.png'
        resultname = f'compare_result-best.{metric_type}.json'

    AGG_COMPARE_DIR = os.path.join(DIRS['PROJECT_DIR'], aggname)

    subproject_labels = glob.glob(os.path.join(DIRS['PROJECT_DIR'],f'{kwargs["label_suffix"]}*'))
    for i, LABEL_DIR in enumerate(subproject_labels):
        PLUGIN_RESULT_DIR = os.path.join(LABEL_DIR, '_plugin.compare_result')
        COMPARE_RESULT_DIR = os.path.join(PLUGIN_RESULT_DIR, resultname)

        label = os.path.basename(LABEL_DIR)
        label = label[len(kwargs["label_suffix"])+1:].split('-')[0]
        with open(COMPARE_RESULT_DIR) as f:
            compare_results = json.load(f)

        # fraction of data points in which all features attributions method agree on the top features 
        # i.e. when endorsement is maximal.
        fraction_agree = compare_results['agree']/compare_results['total']

        y.append(fraction_agree)
        labels.append(label)

    x = list(range(1,1+len(y)))

    font = {'size': 14}
    plt.rc('font', **font)

    plt.figure(figsize=( len(y)-2, 4))  

    plt.gcf().add_subplot(111)
    plt.gca().plot(x, y,linestyle="dashed", marker="v", c='g')
    plt.gca().set_xticks([0.5]+x)
    plt.gca().set_xticklabels([""] + labels, rotation=-60)
    plt.gca().set_title(kwargs['label_suffix'])
    plt.gca().set_ylabel("fraction")
    plt.tight_layout()
    plt.savefig(AGG_COMPARE_DIR)     



