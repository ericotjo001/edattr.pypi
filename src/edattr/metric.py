from .utils import *

def compute_classification_metrics(x,y0):
    # ASSUMPTIONS:
    # both x and y0 are numpy arrays of integers 0,1,2,...,c-1 where
    #   0: negative
    #   1,2,...,c-1: positive 
    # For example 
    #   x[i] == y0[i] == 0, we have TRUE NEGATIVE
    #   x[i] == y0[i] == C for C>0, we have TRUE POSITIVE

    TP_array = ((x>0)*(y0>0)*(x==y0)).astype(int)
    TN_array = ((x==0)*(y0==0)).astype(int)
    FP_array = ((x>0)*(y0==0) + (x>0)*(y0>0)*(x!=y0)).astype(int)
    FN_array = (((x==0)*(y0>0))).astype(int)

    TP = int(np.sum(TP_array))
    TN = int(np.sum(TN_array))
    FP = int(np.sum(FP_array))
    FN = int(np.sum(FN_array))

    acc = np.mean(x==y0)
    recall = TP/(TP+FN) if TP>0 else 0.
    precision = TP/(TP+FP) if TP>0 else 0.
    f1 = 2*recall*precision/(recall+precision) if recall+precision>0 else 0.

    confusion_matrix = {
        'TP':TP,'TN':TN,'FP':FP,'FN':FN,
        'acc': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }
    return confusion_matrix