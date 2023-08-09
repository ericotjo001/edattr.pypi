""" 
python resnets.py --module bottleneck
python resnets.py
python resnets.py --module resnetEmb
"""
import torch
from edattr.model import Bottleneck, eResNet, eResNetEmb1D, make_intermediate_layer_settings_eResNet, make_intermediate_layer_settings_eResNetEmb
from captum.attr import KernelShap, Lime

from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_input(CHANNELS,dim):
    if dim == 1:
        x = torch.randn(size=(4,CHANNELS, 11)).to(device=device)
    elif dim == 2:
        x = torch.randn(size=(4,CHANNELS, 11,17)).to(device=device)
    return x

def test_bottleneck():
    """ Result:
    test_bottleneck
    dim:1  planes: 4            [4, 7, 11]           [4, 16, 11]
    dim:1  planes: 9            [4, 7, 11]           [4, 36, 11]
    dim:1  planes:16            [4, 7, 11]           [4, 64, 11]
    dim:2  planes: 4        [4, 7, 11, 17]       [4, 16, 11, 17]
    dim:2  planes: 9        [4, 7, 11, 17]       [4, 36, 11, 17]
    dim:2  planes:16        [4, 7, 11, 17]       [4, 64, 11, 17]

    Note: the output channels = planes*4
    """
    print('test_bottleneck')

    in_planes = 7
    for dim in [1,2]:
        for planes in [4, 9, 16]:
            bneck = Bottleneck(in_planes, planes, dim=dim).to(device=device)
            x = get_input(in_planes, dim)
            y = bneck(x)

            print(f"dim:{dim}  planes:{'%2s'%(str(planes))}", 
                '%21s'%(str(list(x.shape))), 
                '%21s'%(str(list(y.shape))), 
                )

def test_resnet():
    """
    testing resnets or components...
    test_resnet!
    dim=1 s1 [6922]         [4, 3, 11] ->  [4, 10] | attr:[1, 3, 11] [1, 3, 11]
    dim=1 s2 [26378]        [4, 3, 11] ->  [4, 10] | attr:[1, 3, 11] [1, 3, 11]
    dim=2 s1 [15946]    [4, 3, 11, 17] ->  [4, 10] | attr:[1, 3, 11, 17] [1, 3, 11, 17]
    dim=2 s2 [40010]    [4, 3, 11, 17] ->  [4, 10] | attr:[1, 3, 11, 17] [1, 3, 11, 17]

    """
    print('test_resnet!')

    INPUT_CHANNEL_DIM, OUTPUT_DIM = 3, 10

    s1 = make_intermediate_layer_settings_eResNet(
        planes=[4, 8], # in original implementation, it's [64,128,256,512]
        n_blocks=[2,2], # original resnet18 is like [2,2,2,2], resnet34 is like [3,4,6,3]
        strides=[1,2], # original like [1,2,2,2]
    )

    s2 = make_intermediate_layer_settings_eResNet(
        planes=[4, 8, 16], # in original implementation, it's [64,128,256,512]
        n_blocks=[2,2,3], # original resnet18 is like [2,2,2,2], resnet34 is like [3,4,6,3]
        strides=[1,2,2], # original like [1,2,2,2]
    )

    for dim in [1,2]:
        for ith,iL_settings in enumerate([s1,s2]):
            x = get_input(INPUT_CHANNEL_DIM, dim)     
            resnet = eResNet(INPUT_CHANNEL_DIM, OUTPUT_DIM, iL_settings, dim=dim).to(device=device)
            nparams = count_parameters(resnet)

            kshap = KernelShap(resnet) # we use LIME-based method to compute shap more efficiently
            lime = Lime(resnet)
            
            y = resnet(x)
            y_pred= torch.argmax(y,dim=1)

            resnet.eval() # need this for captum
            i = 0 # let's just test captum on the first item in the batch
            x_ = x[i:i+1] # # but keep dim, so the shape is (1,*)
            attr_kshap = kshap.attribute(x_, target=y_pred[i].item(), n_samples=50)
            attr_lime = lime.attribute(x_, target=y_pred[i].item(), n_samples=50)

            print(f'dim={dim} s{ith+1}', '%-7s'%(f'[{nparams}]'),                 
                '%17s'%(str(list(x.shape))), '->' 
                '%9s'%(str(list(y.shape))), '| attr:'
                '%9s'%(str(list(attr_kshap.shape))),
                '%9s'%(str(list(attr_lime.shape))),
                ) 

def test_resnetemb():
    from mlpemb import MyDataset

    TOKEN_FEATURES = ['gender', 'smoking']
    NUMERICAL_FEATURES = ['score1', 'score2'] 
    TARGET_LABEL_NAME = 'target'
    word_to_ix = {'M':0,'F':1,'yes':2,'no':3}     
    dict_leng = len(word_to_ix)
    C = 3 # no. of classes

    s1, emb1_setting = make_intermediate_layer_settings_eResNetEmb(
        planes=[4, 8], # in original implementation, it's [64,128,256,512]
        n_blocks=[2,2], # original resnet18 is like [2,2,2,2], resnet34 is like [3,4,6,3]
        strides=[1,2], # original like [1,2,2,2]
        nD=5, encoder_out_d=7
        )

    s2, emb2_setting = make_intermediate_layer_settings_eResNetEmb(
        planes=[4, 8, 16], # in original implementation, it's [64,128,256,512]
        n_blocks=[2,2,3], # original resnet18 is like [2,2,2,2], resnet34 is like [3,4,6,3]
        strides=[1,2,2], # original like [1,2,2,2]
        nD=5, encoder_out_d=7
    )    

    dset = MyDataset(TOKEN_FEATURES, NUMERICAL_FEATURES, TARGET_LABEL_NAME)
    dat = DataLoader(dset, batch_size=3)    

    def run_once(ith, iL_settings, emb_setting):      
        resnet = eResNetEmb1D(TOKEN_FEATURES, NUMERICAL_FEATURES, dict_leng, iL_settings, emb_setting, C).to(device=device)
        resnet.eval()
        nparams = count_parameters(resnet)

        kshap = KernelShap(resnet) # we use LIME-based method to compute shap more efficiently
        lime = Lime(resnet)
        
        for i, (x,y0) in enumerate(dat):
            y = resnet(x.to(device=device))
            y_pred= torch.argmax(y,dim=1)

            x_ = x[i:i+1].to(device=device) # # but keep dim, so the shape is (1,*)
            attr_kshap = kshap.attribute(x_, target=y_pred[i].item(), n_samples=50)
            attr_lime = lime.attribute(x_, target=y_pred[i].item(), n_samples=50)

            print(f'dim={1} s{ith+1}', '%-7s'%(f'[{nparams}]'),                 
                '%17s'%(str(list(x.shape))), '->' 
                '%9s'%(str(list(y.shape))), '| attr:'
                '%9s'%(str(list(attr_kshap.shape))),
                '%9s'%(str(list(attr_lime.shape))),
                ) 

    for ith,(iL_settings, emb_setting) in enumerate(zip([s1,s2], [emb1_setting, emb2_setting])):
        print("\n>>>>>", ith)
        run_once(ith, iL_settings, emb_setting)

if __name__ == '__main__':
    print('testing resnets or components...')

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=None)
    parser.add_argument('--module', default='resnet', type=str, help=None)
    args, unknown = parser.parse_known_args()

    if args.module == 'bottleneck':
        test_bottleneck()
    elif args.module == 'resnet':
        test_resnet()
    elif args.module == 'resnetEmb':
        test_resnetemb()
    else:
        raise Exception('unknown module')

