""" 
python mlpemb.py
python mlpemb.py --type 1
python mlpemb.py --type 2
python mlpemb.py --type 3

Output is like:

test mlp+emb
===== type:2 =====
n params: 738
x.shape:[3, 4] y.shape:[3, 3]
 attr_kshap/lime.shapes:[1, 4],[1, 4]
 attr_kshap/lime.shapes:[1, 4],[1, 4]
 attr_kshap/lime.shapes:[1, 4],[1, 4]
x.shape:[2, 4] y.shape:[2, 3]
 attr_kshap/lime.shapes:[1, 4],[1, 4]
 attr_kshap/lime.shapes:[1, 4],[1, 4]
"""

import numpy as np
import pandas as pd

import torch
from edattr.model import MLPEmb
from captum.attr import KernelShap, Lime

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME):
        super(MyDataset, self).__init__()
        self.df = pd.DataFrame({
            'gender': [0,1,0,1,0],
            'smoking': [2,3,2,2,3],
            'score1': [1.0,1.1,1.2, 2.3, 1.8],
            'score2': [9.0,9.1, 9.77 , 8.2, 10.8],
            'target': [0,1,2,0,1]
        })        
        self.TOKEN_FEATURES = TOKEN_FEATURES
        self.NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.TARGET_LABEL_NAME = TARGET_LABEL_NAME

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        tokens = self.df[self.TOKEN_FEATURES].loc[i].to_numpy()
        numerics = self.df[self.NUMERICAL_FEATURES].loc[i].to_numpy()

        y0 = self.df[self.TARGET_LABEL_NAME].loc[i]
        return np.concatenate((tokens, numerics)), y0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_mlb(fc=[7,3], encoder_out_d=7, nD=5):
    TOKEN_FEATURES = ['gender', 'smoking']
    NUMERICAL_FEATURES = ['score1', 'score2'] 
    TARGET_LABEL_NAME = 'target'
    word_to_ix = {'M':0,'F':1,'yes':2,'no':3}
    dict_leng = len(word_to_ix)

    C = 3
    layers = {
        'nD': nD,
        'encoder_out_d': encoder_out_d,
        'fc': fc,
    }
    model = MLPEmb(layers, TOKEN_FEATURES, NUMERICAL_FEATURES, dict_leng)
    model.eval()
    print('n params:', count_parameters(model))

    dset = MyDataset(TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME)
    dat = DataLoader(dset, batch_size=3)

    ak = KernelShap(model)
    al = Lime(model)

    for i, (x,y0) in enumerate(dat):
        # print(x)
        # print('y0:', y0)
        """ Like
        tensor([[0.0000, 2.0000, 1.0000, 9.0000],
            [1.0000, 3.0000, 1.1000, 9.1000],
            [0.0000, 2.0000, 1.2000, 9.7700]], dtype=torch.float64)
        y0: tensor([0, 1, 2])
        """

        y = model(x)
        print(f'x.shape:{list(x.shape)} y.shape:{list(y.shape)}',)

        y_pred = torch.argmax(y,dim=1)
        b = x.shape[0] # batch size
        for i in range(b):
            # Make sure to feed input one by one (not by batches)
            x_ = x[i:i+1] # but keep dim, so the shape is (1,*) 

            target = y_pred[i].item()
            attr_kshap = ak.attribute(x_, target=target)
            attr_lime = al.attribute(x_, target=target) 

            aks = list(attr_kshap.shape)
            als = list(attr_lime.shape)
            print(f' attr_kshap/lime.shapes:{aks},{als}')


if __name__ == '__main__':
    print('test mlp+emb')
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=None)
    parser.add_argument('--type', default=0, type=int, help=None)
    args, unknown = parser.parse_known_args()
    kwargs = vars(args)

    print(f"===== type:{kwargs['type']} =====")
    if kwargs['type'] == 0:
        setting = {'fc':[7,3], 'encoder_out_d': 7, 'nD':5}
    elif kwargs['type'] == 1:
        setting = {'fc':[7,3], 'encoder_out_d': 14, 'nD':5}
    elif kwargs['type'] == 2:
        setting = {'fc':[7,14,3], 'encoder_out_d': 14, 'nD':14}
    elif kwargs['type'] == 3:
        setting = {'fc':[7,14,3], 'encoder_out_d': 14, 'nD':21}
    else:
        raise NotImplementedError()

    test_mlb(**setting)