###################################################
# Instruction Manual 
###################################################
python main.py --mode preprocess_normalize

# run it one more time after you set TOKEN_FEATURES and NUMERICAL_FEATURES
python main.py --mode preprocess_normalize

python main.py --mode standard --label toy3_standard_mlp_0000-0
python main.py --mode standard --label toy3_standard_mlp_0001-0
python main.py --mode standard --label toy3_standard_mlp_0002-0
python main.py --mode standard --label toy3_standard_resnet_0000-0
python main.py --mode standard --label toy3_standard_resnet_0001-0
python main.py --mode standard --label toy3_standard_resnet_0002-0
python main.py --mode standard --label toy3_standard_transformer_0000-0
python main.py --mode standard --label toy3_standard_transformer_0001-0
python main.py --mode standard --label toy3_standard_transformer_0002-0
python main.py --mode standard_aggregate


###################################################
# Instruction Manual (DEV and DEBUG version)
################################################### 
python main.py --mode preprocess_normalize
python main.py --mode preprocess_normalize

python main.py --mode standard --label toy3_standard_mlp_0000-123

python main.py --mode standard --label toy3_standard_mlp_0000-123 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_mlp_0001-124 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_mlp_0002-125 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_resnet_0000-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_resnet_0001-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_resnet_0002-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_transformer_0000-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_transformer_0001-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
python main.py --mode standard --label toy3_standard_transformer_0002-999 --n_epochs 2 --eec_n_epochs 4  --toggle 1111
