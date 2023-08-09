"""
python endorsement_feature_view.py
python endorsement_feature_view.py --n_features 21
python endorsement_feature_view.py --n_features 37
python endorsement_feature_view.py --n_features 99
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def toy_endorsement_data(n_data=1000,n_votes=6, n_features=7):
    nhalf = int(n_features/2)
    p = np.array([i for i in range(n_features)])[::-1]
    p = p/np.sum(p)

    kfold_results = {}
    for idx_ in range(n_data):
        endorsement = {i:0 for i in range(n_features)}
        votes = np.random.choice(range(n_features), replace=True, size=(6,),p=p)
        for v in votes:
            endorsement[v] += 1
        endorsement = {feature_idx:v for feature_idx,v in endorsement.items() if v>0}

        ends = {
            'endorsement': endorsement,
            'isCorrect': np.random.choice([False,True], p=[0.8,0.2]), 
            'y0': np.random.choice([0,1,2], p=[0.5,0.25, 0.25]),
        }
        kfold_results[idx_] = ends
    return kfold_results

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--n_features', default=7, type=int, help=None) 

    args, unknown = parser.parse_known_args()
    kwargs = vars(args)  # is a dictionary

    from edattr.factory import get_home_path
    HOME_ = get_home_path() # Desktop or maybe user/home, depending on your OS

    WORKSPACE_DIR =  os.path.join(HOME_, "Desktop", "edattr.ws")
    TEST_FOLDER_DIR = os.path.join(WORKSPACE_DIR, '_tests')
    os.makedirs(TEST_FOLDER_DIR,exist_ok=True)

    n_votes=6
    n_features=kwargs['n_features']
    kfold_results = toy_endorsement_data(n_votes=n_votes, n_features=n_features)
    for idx_, ends in kfold_results.items():
        assert(np.sum([v for i,v in ends['endorsement'].items()])==n_votes)

    from edattr.endorse import kFoldClassifierEndorsementVis
    vis = kFoldClassifierEndorsementVis(DIRS={})

    features = [f"features-long-name-{i}" for i in range(n_features)]
    FEATURE_VIEW_SUFFIX = os.path.join(TEST_FOLDER_DIR, f'featureView-{n_features}-')
    vis.build_endorsement_feature_view(features, kfold_results, FEATURE_VIEW_SUFFIX,print_save_dir=True)
