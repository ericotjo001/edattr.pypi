import os, joblib, time, json, shutil, glob, copy, itertools, re
import string, random
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn

from sklearn.cluster import BisectingKMeans

def numpy_array_to_df(df_, features):
    df_processed = {}
    for i,feature in enumerate(features):
        df_processed[feature] = df_[:,i]
    df_processed = pd.DataFrame(df_processed)
    return df_processed

def average_every_n(vals,iters=None, n=10):
    n_excess = len(vals)%n
    n_av = len(vals)//n

    last_round_index = int(n_av*n)
    if iters is None:
        iters = np.array(range(1,1+len(vals)))
    else:
        assert(len(iters)==len(vals))
    t1 = np.array(iters[:last_round_index]).reshape(n_av,n)
    y1 = np.array(vals[:last_round_index]).reshape(n_av,n)

    t1 = list(t1[:,-1])
    y1 = list(np.mean(y1,axis=1))

    if n_excess>0:
        t1.append(iters[-1] )
        y1.append(np.mean(vals[last_round_index:]))
    return t1, y1

def get_uniform_hist_bin(data, binwidth=1, low_end=None, high_end=None):
    low_end = min(np.floor(data)) if low_end is None else low_end
    high_end = max(np.ceil(np.array(data)*1.25))  if high_end is None else high_end
    return np.arange(low_end, high_end + binwidth, binwidth)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sort_dictionary_by_values_desc(d):
    dsorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return dsorted


UMSG_IMPLEMENT_DOWNSTREAM = 'Util msg: implement this function downstream'


