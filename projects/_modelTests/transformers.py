"""
python transformers.py
python transformers.py --mode emb
"""
import torch
from edattr.model import eTFClassifier
from captum.attr import KernelShap, Lime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_transformer():
    """
    testing transformer-based models...
    nhead=4 n_enc=1 dim_ff=128 [11583]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=4 n_enc=1 dim_ff=256 [18879]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=4 n_enc=2 dim_ff=128 [22267]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=4 n_enc=2 dim_ff=256 [36859]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=8 n_enc=1 dim_ff=128 [13731]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=8 n_enc=1 dim_ff=256 [22051]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=8 n_enc=2 dim_ff=128 [26435]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    nhead=8 n_enc=2 dim_ff=256 [43075]     [4, 24] ->   [4, 3] | attr:  [1, 24]   [1, 24]
    """

    INPUT_DIMENSION = 24
    OUT_DIMENSION = 3

    for nhead in [4,8]:
        for n_enc in [1,2]:
            for dim_ff in [128,256]:
                x = torch.rand((4, INPUT_DIMENSION))
                transformer_model = eTFClassifier(INPUT_DIMENSION, OUT_DIMENSION, nhead=nhead, n_enc=n_enc, dim_ff=dim_ff)
                nparams = count_parameters(transformer_model)
                y = transformer_model(x)

                kshap = KernelShap(transformer_model) # we use LIME-based method to compute shap more efficiently
                lime = Lime(transformer_model)    

                y_pred= torch.argmax(y,dim=1)

                transformer_model.eval() # need this for captum
                i = 0 # let's just test captum on the first item in the batch
                x_ = x[i:i+1] # # but keep dim, so the shape is (1,*)
                attr_kshap = kshap.attribute(x_, target=y_pred[i].item(), n_samples=50)
                attr_lime = lime.attribute(x_, target=y_pred[i].item(), n_samples=50)

                print(f'nhead={nhead} n_enc={n_enc} dim_ff={dim_ff}', '%-7s'%(f'[{nparams}]'),
                    '%11s'%(str(list(x.shape))), '->' 
                    '%9s'%(str(list(y.shape))), '| attr:'
                    '%9s'%(str(list(attr_kshap.shape))),
                    '%9s'%(str(list(attr_lime.shape))),
                    ) 

def test_transformer_emb():
    """
    testing transformer-based models...
    x.shape:[3, 4] y.shape:[3, 3]
    nhead=4 n_enc=1 nD=64 [37599]      [3, 4] ->   [3, 3] | attr:   [1, 4]    [1, 4]
    x.shape:[3, 4] y.shape:[3, 3]
    nhead=4 n_enc=1 nD=128 [106655]      [3, 4] ->   [3, 3] | attr:   [1, 4]    [1, 4]
    x.shape:[3, 4] y.shape:[3, 3]
    nhead=7 n_enc=1 nD=128 [107994]      [3, 4] ->   [3, 3] | attr:   [1, 4]    [1, 4]
    x.shape:[3, 4] y.shape:[3, 3]
    nhead=7 n_enc=2 nD=128 [214123]      [3, 4] ->   [3, 3] | attr:   [1, 4]    [1, 4]
    
    """
    TOKEN_FEATURES = ['gender', 'smoking']
    NUMERICAL_FEATURES = ['score1', 'score2'] 
    TARGET_LABEL_NAME = 'target'
    word_to_ix = {'M':0,'F':1,'yes':2,'no':3}
    dict_leng = len(word_to_ix)
    C = 3

    from mlpemb import MyDataset 
    from torch.utils.data import DataLoader
    dset = MyDataset(TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME)
    dat = DataLoader(dset, batch_size=3)

    from edattr.model import eTFClassifierEmb

    for nhead, n_enc, nD in zip([4,4,7,7],[1,1,1,2],[64,128,128, 128]):
        model = eTFClassifierEmb(TOKEN_FEATURES, NUMERICAL_FEATURES, dict_leng, C, nD=nD, nhead=nhead, n_enc=n_enc,)
        model.eval()

        for i, (x,y0) in enumerate(dat):
            nparams = count_parameters(model)

            y = model(x)
            print(f'x.shape:{list(x.shape)} y.shape:{list(y.shape)}',)

            kshap = KernelShap(model) # we use LIME-based method to compute shap more efficiently
            lime = Lime(model)    

            y_pred= torch.argmax(y,dim=1)

            model.eval() # need this for captum
            i = 0 # let's just test captum on the first item in the batch
            x_ = x[i:i+1] # # but keep dim, so the shape is (1,*)
            attr_kshap = kshap.attribute(x_, target=y_pred[i].item(), n_samples=50)
            attr_lime = lime.attribute(x_, target=y_pred[i].item(), n_samples=50)

            print(f'nhead={nhead} n_enc={n_enc} nD={nD}', '%-7s'%(f'[{nparams}]'),
                '%11s'%(str(list(x.shape))), '->' 
                '%9s'%(str(list(y.shape))), '| attr:'
                '%9s'%(str(list(attr_kshap.shape))),
                '%9s'%(str(list(attr_lime.shape))),
                ) 

            break # just do see the output for 1 batch

if __name__ == '__main__':
    print('testing transformer-based models...')
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=None)
    parser.add_argument('--mode', default=None, type=str, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)

    if kwargs['mode'] is None:
        test_transformer()
    elif kwargs['mode'] == 'emb':
        test_transformer_emb()
