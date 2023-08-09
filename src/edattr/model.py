from .utils import *

"""
We reimplement some models for several reasons, like:
1. compatibility with pytorch captum functions
2. standardization for edattr use
3. simplification of initiations. Yes, sometimes we don't need the full resnets for smaller problems, so we're gonna allow users to easily set fewer numbers of blocks
"""



#################################################
#                                               #
#             CLASSIFICATION MODELS             #
#                                               #
#################################################


####################################
#             MLP
####################################
# For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()

        self.layers = layers

        self.fc_modules = nn.ModuleDict({})
        self.activations = nn.ModuleDict({})
        for i in range(len(self.layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.kaiming_normal_(layer.weight.data, 
                mode='fan_in', nonlinearity='leaky_relu')
            self.fc_modules.update({'fc'+ str(i) : layer})
            self.activations.update({'act'+str(i): nn.LeakyReLU()})

    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = self.fc_modules['fc'+ str(i)](x)
            x = self.activations['act'+str(i)](x)
        return x

####################################
#             Resnets 
####################################
""" 
Our model is called eResNet, but it's just a variant of Resnets in terms of number of blocks.
For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)

No special meaning in the "e" beyond the fact that we're talking about "endorsement" method for XAI.

Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
Quick References:

resnet18: _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
resnet34: _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
resnet50: _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
    weights: Optional[WeightsEnum], progress: bool, **kwargs: Any, ) -> ResNet: 
    ...
    model = ResNet(block, layers, **kwargs)
    ...

class ResNet(nn.Module):
    def __init__( self,
        block: Type[Union[BasicBlock, Bottleneck]], 
        layers: List[int], 
        num_classes: int = 1000, ...):

        ...
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=...)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=...)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=...)        
        ...

def _make_layer(self,
    block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
    stride: int = 1, dilate: bool = False,) -> nn.Sequential:   

def conv3x3():
    ...
    inplanes = 64
    ...
"""

class eResNetHead(nn.Module):
    def __init__(self, INPUT_CHANNEL_DIM, inplanes, norm_layer=None, dim=2):
        super(eResNetHead, self).__init__()
        self.dim=dim
        self.inplanes = inplanes
        self.INPUT_CHANNEL_DIM = INPUT_CHANNEL_DIM

        if norm_layer is None:
            norm_layer = get_default_norm_layer(dim)        
        self.norm_layer = norm_layer

        if self.dim==2:
            self._setup_head_modules_dim2()
        elif self.dim==1:
            self._setup_head_modules_dim1()

    def _setup_head_modules_dim2(self):
        self.head_modules = nn.ModuleDict({
            'conv1': nn.Conv2d(self.INPUT_CHANNEL_DIM, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            'bn1': self.norm_layer(self.inplanes),
            'relu': nn.ReLU(),
            'maxpool': nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            })

    def _setup_head_modules_dim1(self):
        self.head_modules = nn.ModuleDict({
            'conv1': nn.Conv1d(self.INPUT_CHANNEL_DIM, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            'bn1': self.norm_layer(self.inplanes),
            'relu': nn.ReLU(),
            'maxpool': nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            })

    def forward(self,x):
        for modname in self.head_modules:
            x = self.head_modules[modname](x)
        return x

def make_intermediate_layer_settings_eResNet(planes, n_blocks, strides):
    ilayer_setting = [
        {'planes': p, 'n_blocks': n_, 'stride': s} for p,n_,s in zip(planes, n_blocks, strides)
    ]
    return ilayer_setting

def _setup_out_modules(last_conv_plane, expansion, OUTPUT_DIM, dim=2):
    out_modules = nn.ModuleDict({})
    if dim==2:
        out_modules.update({'avgpool':nn.AdaptiveAvgPool2d((1, 1))})
    if dim==1:
        out_modules.update({'avgpool':nn.AdaptiveAvgPool1d((1,))})

    out_modules.update({'fc': nn.Linear(last_conv_plane * expansion, OUTPUT_DIM)})
    return out_modules

class eResNet(nn.Module):
    def __init__(self, INPUT_CHANNEL_DIM, OUTPUT_DIM, intermediate_layer_settings, dim=2, norm_layer=None, input_type=None):
        super(eResNet, self).__init__()
        if norm_layer is None:
            norm_layer = get_default_norm_layer(dim)
        
        self.INPUT_CHANNEL_DIM = INPUT_CHANNEL_DIM # channels like rgb
        self.OUTPUT_DIM = OUTPUT_DIM
        self.dim = dim # 1 for 1D data like time series, 2 for images
        self.norm_layer = norm_layer

        self.layer_settings = intermediate_layer_settings
        self.n_mid_layers = len(self.layer_settings)

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # smaller resnets use BasicBlock, but let's use the Bottleneck uniformly
        self.block = Bottleneck 

        self.setup_head_modules()

        self.layers = nn.ModuleDict({})
        self.setup_layers() 

        self.setup_out_modules()

        for m in self.modules():
            # don't worry, it will iterate everything down all the way to the contents of ModuleDict
            if isinstance(m, (nn.Conv2d,nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # extra adhoc settings
        self.input_type = input_type

    def setup_head_modules(self):
        self.head_modules = eResNetHead(self.INPUT_CHANNEL_DIM, self.inplanes, dim=self.dim)

    def setup_out_modules(self):
        last_iL_settings = self.layer_settings[-1]
        last_conv_plane = last_iL_settings['planes'] 
        self.out_modules = _setup_out_modules(last_conv_plane, 
            self.block.expansion, self.OUTPUT_DIM, dim=self.dim)

    def setup_layers(self):
        for i,layer_setting in enumerate(self.layer_settings):
            self.update_layers(str(i+1),layer_setting)

    def update_layers(self,L,layer_setting):
        assert(type(L)==type('yeah, L must be a string.'))
        # layer_setting
        block = self.block
        planes = layer_setting['planes']
        blocks = layer_setting['n_blocks']
        stride = layer_setting['stride']
        self.layers.update(
            {L: self._make_layer(L, block, planes, blocks, stride=stride)}
        )

    def _make_layer(self, L, block, planes, blocks, stride=1):
        # block: Type[Union[BasicBlock, Bottleneck]],
        # blocks: int

        # for now, dilate is always False (the option is available in original implementation)

        norm_layer = self.norm_layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = get_downsample(self.inplanes, planes, block.expansion, 
                stride, norm_layer, dim=self.dim)

        previous_dilation = self.dilation

        layers = []
        layers.append(
            block(self.inplanes, planes, dim=self.dim, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer)
            )
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dim=self.dim, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,)
            )
        return nn.Sequential(*layers)

    def forward(self,x):
        if self.input_type=='single_flat_channel':
            x = x.unsqueeze(1)

        x = self.head_modules(x)
        for i in range(self.n_mid_layers):
            L = str(i+1)
            x = self.layers[L](x)

        x = self.out_modules['avgpool'](x)
        x = torch.flatten(x, 1) # flatten to 1 dim tensor, starting from start_dim=1
        x = self.out_modules['fc'](x)
        return x


def conv1x1(in_planes, out_planes, stride=1, dim=2):
    """1x1 convolution"""
    if dim==2:
        conv = nn.Conv2d
    elif dim==1:
        conv = nn.Conv1d
    return conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dim=2):
    """3x3 convolution with padding"""
    if dim==2:
        conv = nn.Conv2d
    elif dim==1:
        conv = nn.Conv1d
    return conv(in_planes, out_planes,
        kernel_size=3, stride=stride, padding=dilation, groups=groups,bias=False,dilation=dilation
    )

def get_downsample(inplanes, planes, expansion, stride, norm_layer, dim=2):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes * expansion, stride, dim=dim),
        norm_layer(planes * expansion),
    )
    return downsample

def get_default_norm_layer(dim):
    if dim==2:
        norm_layer = nn.BatchNorm2d
    elif dim==1:
        norm_layer = nn.BatchNorm1d
    return norm_layer

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, dim=2,
        stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = get_default_norm_layer(dim)

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width, dim=dim)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(width, width, stride, groups, dilation, dim=dim)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU() # yes, we create another relu for captum compatibility. Don't use inplace=True too

        self.conv3 = conv1x1(width, planes * self.expansion, dim=dim)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = get_downsample(inplanes, planes, self.expansion, stride, norm_layer, dim=dim)
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out        

####################################
#             Transformer 
####################################
"""
Our model is called eTFClassifier for e-Transformer-classifer
# For NAIVE NUMERICAL DATAFRAME (see data.py, Data Frame Types)

eTransformer is just a variant of transformer model in terms of number of blocks, heads, attentions.
As before, no special meaning in the "e" beyond the fact that we're talking about "endorsement" method for XAI.
"""

class eTransformer(nn.Transformer):
    def __init__(self, **kwargs):
        super(eTransformer, self).__init__(**kwargs)

    def forward(self, src):
        # for classification, the forward function doesn't need the encoder, so we overwrite this function
        """ assume `batch_first=True`.
            - src: `(N, S, E)`       
        """

        is_batched = src.dim() == 3
        out = self.encoder(src, mask=None, src_key_padding_mask=None)
        return out

import torch.nn.functional as F
class eTFClassifier(nn.Module):
    def __init__(self, INPUT_DIMENSION, OUT_DIMENSION, nhead=7, n_enc=1, dim_ff=128):
        super(eTFClassifier, self).__init__()
        
        # dim_padded: this is approximately INPUT_DIMENSION. 
        # We need to make sure that d_model%nhead==0 in self.tf, that's why add the fc_reshape for "padding"
        dim_padded = INPUT_DIMENSION + (nhead-INPUT_DIMENSION%nhead)
        assert(dim_padded%nhead==0) 

        self.fc_reshape = nn.Linear(INPUT_DIMENSION, dim_padded)

        self.tf = eTransformer(
            d_model=dim_padded, # pytorch default=512 
            nhead=nhead,  # pytorch default=8
            num_encoder_layers=n_enc, # pytorch default=6
            num_decoder_layers=0, # pytorch default=6. We will use simple Linear decoder
            dim_feedforward=dim_ff, # pytorch default=2048
            dropout=0.1,
            activation= F.relu,
            layer_norm_eps=1e-5, 
            batch_first=True, norm_first= False,
            device=None, dtype=None
        )
        # note: nn.Transformer input is like (batch_size, CONTEXT_SIZE, embedding dim) in NLP. CONTEXT_SIZE is token length basically
        # self.tf.decoder will be an empty ModuleList() with norm layer (won't be used)

        self.fc = nn.Linear(dim_padded,OUT_DIMENSION) # final decoder

    def forward(self,x):
        # x is like (batch_size, INPUT_DIMENSION)
        x = self.fc_reshape(x).unsqueeze(1) # like (batch_size, 1, dim_padded)
        x = self.tf(x) # like (batch_size, 1, dim_padded)
        x = self.fc(x.squeeze(1))
        return x


####################################
#         MLP + Emb
####################################
# For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)

class MLPEmb(nn.Module):
    """
    MLP whose input is the standard tensor plus NLP-like tokens
    Yes, Emb is for embedding 
    """
    def __init__(self, layers, token_features, numerical_features, dict_leng):
        """
        # example:
        token_features = ["gender","smoke"]
        numerical_features = ["score", "height", "weight"]

        C = 3 # no. of classes
        layers = {
            'nD':5, # dim of embedding
            'encoder_out_d': 7,
            'fc': [7,C], # no. of neurons in the ouput fc
        }   
        """
        super(MLPEmb, self).__init__()
        self.layers = L = layers

        self.token_features = token_features
        self.numerical_features = numerical_features
        self.n_tf = self.token_length = len(token_features)
        self.n_nf = len(numerical_features)

        # embedding + simple encoder for embedding
        self.emb = nn.Embedding(dict_leng, L['nD'])
        self.encoder = nn.Conv1d(L['nD'], L['encoder_out_d'], self.token_length) 

        fc_dim = L['encoder_out_d'] + self.n_nf

        fcs = []
        fclayers = L['fc']
        for i,nneuron in enumerate(fclayers):        
            din = fc_dim if i==0 else fclayers[i-1]
            dout = fclayers[i]
            fcs.append(nn.Linear(din, dout))
        self.fcs = nn.Sequential(*fcs) # nn.Linear(fc_dim, layers['out'])
        
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",)

    def forward(self, x):
        xt = torch.round(x[:, :self.n_tf]).to(torch.long) # rounding up or down will happen here! 
        xn = x[:, self.n_tf:].to(torch.float)
        # xt like (batch_size, n_tf=no of tokens )
        #   xt contains tokens (integer form) for embedding
        # xn like (batch_size, d)
        # 
        # !!Warning!! x may not be naturally in the above form concat(xt,xn)  <-- the order matters!
        #   Columns that contain tokens and numerics are not usually ordered in any specific way.
        #   During the definition  of Dataset, make sure to convert it to this form. 
        #
        # !!Warning 2!! Why rounding up or down? In our implementation, we use KMeans to cluster things
        #   However, xt are tokens converted into int/long and kmeans will yield floating point numbers
        #   during our EEC process. You can see these decimal numbers
        #   in files like eec-train-data-t3.csv inside eec.result folder

        tokens_ = self.emb(xt)
        # print(tokens_.shape) # like (batch_size, no. of tokens, nD)
        #   e.g. torch.Size([4, 2, 5]). 
        # Note: no. of tokens can be variable like 
        #   no. of words in a sentence!

        t_ = self.encoder(torch.transpose(tokens_,1,2))
        # print(t_.shape) 
        # (batch_size, encoder_out_d, *) 
        #   like torch.Size([4, encoder_out_d, 1])

        x = self.fcs(torch.cat((torch.mean(t_,axis=-1), xn), axis=-1))
        return x                        

####################################
#         ResNet + Emb
####################################
# For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)

def make_intermediate_layer_settings_eResNetEmb(planes, n_blocks, strides, nD, encoder_out_d):
    ilayer_setting = [
        {
            'planes': p, 
            'n_blocks': n_, 
            'stride': s,
        } for p,n_,s in zip(planes, n_blocks, strides, )
    ]
    emb_setting = {'nD': nD, 'encoder_out_d': encoder_out_d}
    return ilayer_setting, emb_setting

class eResNetEmb1D(eResNet):
    def __init__(self, token_features, numerical_features, dict_leng, intermediate_layer_settings, emb_setting, out_dim, norm_layer=None, input_type=None):

        self.token_features = token_features
        self.numerical_features = numerical_features
        self.n_tf = self.token_length = len(token_features)
        self.n_nf = len(numerical_features)

        INPUT_CHANNEL_DIM = self.n_nf + emb_setting['nD']
        OUTPUT_DIM = out_dim 
        super(eResNetEmb1D, self).__init__(INPUT_CHANNEL_DIM, OUTPUT_DIM, intermediate_layer_settings, dim=1, norm_layer=norm_layer, input_type=input_type)

        self.emb_setting = emb_setting
        self.emb = nn.Embedding(dict_leng, emb_setting['nD'])
        self.fc_head = nn.Linear(self.n_nf, self.n_nf*self.n_tf)

    def forward(self, x):
        xt = torch.round(x[:, :self.n_tf]).to(torch.long) # rounding up or down will happen here! 
        xn = x[:, self.n_tf:].to(torch.float)
        # print(xt.shape) # (batch_size, self.token_length)
        # print(xn.shape) # (batch_size, self.n_nf)

        tokens_ = self.emb(xt)
        xn = self.fc_head(xn).reshape(-1,self.token_length, self.n_nf)
        # print(tokens_.shape) # (batch_size, self.token_length, nD)
        # print(xn.shape) # (batch_size,self.token_length, self.n_nf)

        x = torch.cat((tokens_,xn), axis=2).transpose(1,2)
        # print(x.shape, )  # (batch_size, self.n_nf + nD, self.token_length)

        x = self.head_modules(x)
        for i in range(self.n_mid_layers):
            L = str(i+1)
            x = self.layers[L](x)

        x = self.out_modules['avgpool'](x)
        x = torch.flatten(x, 1) # flatten to 1 dim tensor, starting from start_dim=1
        x = self.out_modules['fc'](x)
        return x

####################################
#      Transformer + Embedding
####################################
# For TokenAndFloat DATAFRAME (see data.py, Data Frame Types)


class eTFClassifierEmb(nn.Module):
    def __init__(self, token_features, numerical_features, dict_leng, C, nD=128, nhead=7, n_enc=1, dim_ff=128, ):
        super(eTFClassifierEmb, self).__init__()

        self.token_features = token_features
        self.numerical_features = numerical_features
        self.n_tf = self.token_length = len(token_features)
        self.n_nf = len(numerical_features)

        dim_padded = nD + (nhead-nD%nhead)

        # numerical part
        self.fc_num = nn.Linear(self.n_nf, dim_padded)
        # token part
        self.emb = nn.Embedding(dict_leng, dim_padded)

        # transformer!
        self.tf = eTransformer(
            d_model=dim_padded, # pytorch default=512 
            nhead=nhead,  # pytorch default=8
            num_encoder_layers=n_enc, # pytorch default=6
            num_decoder_layers=0, # pytorch default=6. We will use simple Linear decoder
            dim_feedforward=dim_ff, # pytorch default=2048
            dropout=0.1,
            activation= F.relu,
            layer_norm_eps=1e-5, 
            batch_first=True, norm_first= False,
            device=None, dtype=None
        )        
        self.fc = nn.Linear(dim_padded, C)

    def forward(self, x):
        xt = torch.round(x[:, :self.n_tf]).to(torch.long) # rounding up or down will happen here! 
        xn = x[:, self.n_tf:].to(torch.float)
        # print(xt.shape) # (batch_size, self.token_length)
        # print(xn.shape) # (batch_size, self.n_nf)

        tokens_ = self.emb(xt)
        xn = self.fc_num(xn).unsqueeze(1)

        x = torch.cat((tokens_, xn), axis=1)
        # print(x.shape) # (batch, adjusted token length, nD)

        x = self.tf(x)
        x = torch.mean(x, axis=1) # mean along the token length
        
        x = self.fc(x)
        return x