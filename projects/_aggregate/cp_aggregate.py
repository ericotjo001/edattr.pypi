"""
Welcome! cp_aggregate stands for CROSS PROJECT aggregate.
We wanna plot our results across different projects, just some fancy display utility, if you like.
"""

import os, joblib, json, glob
import numpy as np
import matplotlib.pyplot as plt

def get_checkpoint_dirs(CKPT_DIR):
    if CKPT_DIR == 'auto':
        from edattr.factory import manage_dirs
        DIRS = manage_dirs(**{
            'WORKSPACE_DIR':None,
            'DIRECTORY_MODE': 'bypass'
            })
        WORKSPACE_DIR = DIRS['WORKSPACE_DIR']
        CKPT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoint')         
    return CKPT_DIR

def test_result_by_labels(parser):
    parser.add_argument('--LABELS_DIR', default="labels.json", type=str, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    with open(kwargs['LABELS_DIR']) as f:
        labels = json.load(f)

    # this is where the project results are stored
    CKPT_DIR = get_checkpoint_dirs(labels['CKPT_DIR'])
    AGGREGATE_RESULT_FOLDER = os.path.join(CKPT_DIR,'_cp_aggregate')
    os.makedirs(AGGREGATE_RESULT_FOLDER, exist_ok=True) 

    metrics = ["acc", "recall", "precision", "f1"]
    plot_results = {m:{} for m in metrics} 
    plot_results_cml = {m:{} for m in metrics}
    for project, label_names in labels["projects"].items():
        project_short_name = labels["PROXY_NAMES"][project]

        for label_name in label_names:
            RESULT_DIR = os.path.join(CKPT_DIR, project, label_name, "test_result")
            # print(RESULT_DIR)
            for result_dir in glob.glob(os.path.join(RESULT_DIR, 'test-output.*.data')):
                # print('  >>',os.path.basename(result_dir))
                test_result = joblib.load(result_dir)
                # print(test_result)

                for this_best_model, results in test_result.items():
                    # print(this_best_model, results["acc"])
                    for metric in metrics:
                        if project_short_name not in plot_results[metric]:
                            plot_results[metric][project_short_name] = []
                        plot_results[metric][project_short_name].append(results[metric])

            ########## Common ML results, if exist #############
            CML_RESULT_DIR = os.path.join(RESULT_DIR,'common_ml_results.json')
            with open(CML_RESULT_DIR) as f:
                cml_results = json.load(f)
            for model_type, result_by_model in cml_results.items():
                # model type is like k-4-best.acc, the NN model we compare to
                for common_ml, cml_result in result_by_model.items():
                    for metric in metrics:
                        if project_short_name not in plot_results_cml[metric]:
                            plot_results_cml[metric][project_short_name] = []
                        plot_results_cml[metric][project_short_name].append(cml_result[metric])
            #####################################################

    scale = 1.5
    bpcol = (0.27,0.27,0.77,0.77)
    plt.rc('font', **{'size':17})
    for i,m in enumerate(plot_results):
        # m is metric
        horizontal_labels = ['']
        values = []
        for project_short_name, val in plot_results[m].items():
            horizontal_labels.append(project_short_name)
            values.append(val)

        fig_dir = os.path.join(AGGREGATE_RESULT_FOLDER, f'cp-{m}.png')
        plt.figure(figsize=(int(len(horizontal_labels)*scale),int(6*scale)))

        for j,val in enumerate(values):
            tmpx_ = np.random.normal(j+1,0.1,size=(len(val,)))
            plt.gca().scatter(tmpx_, val, alpha=0.1, edgecolor=None)
        
        for j, (project_short_name, val) in enumerate(plot_results_cml[m].items()):
            tmpx_ = np.random.normal(j+1,0.1,size=(len(val,)))
            plt.gca().scatter(tmpx_, val, alpha=0.57, marker='x')

        plt.gca().boxplot(values, 
            boxprops={'color': bpcol},
            whiskerprops={'color':bpcol},
            capprops={'color':bpcol},
            flierprops={'markerfacecolor':'none', 'markeredgecolor':(1,0,0,0.9),'markersize':3})
        plt.gca().set_xticks(range(len(horizontal_labels)), horizontal_labels, rotation=30)
        plt.gca().set_xlim([0.5,None])
        plt.gca().set_ylim([-0.1,1.05])
        plt.gca().set_title(m)
        plt.tight_layout()
        plt.savefig(fig_dir)
        print(f'figures saved to {fig_dir}')
    
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)
    parser.add_argument('-m','--mode', default=None, type=str, help=None)
    # parser.add_argument('--id', nargs='+', default=['a','b']) # for list args

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary
    
    
    if kwargs['mode'] == 'test_result_by_labels':
        test_result_by_labels(parser)
    else:
        print('invalid mode?')
