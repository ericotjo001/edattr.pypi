######################################################
# Instruction Manual 
######################################################

# run preprocessing twice. The first to set TOKEN_FEATURES and NUMERICAL_FEATURES
python main.py --mode preprocess_normalize
python main.py --mode preprocess_normalize

python main.py --mode kfold --label maternalhealth_kfold_mlp_0000-0 
python main.py --mode kfold --label maternalhealth_kfold_mlp_0001-0 
python main.py --mode kfold --label maternalhealth_kfold_mlp_0002-0  
python main.py --mode kfold --label maternalhealth_kfold_resnet_0000-0 
python main.py --mode kfold --label maternalhealth_kfold_resnet_0001-0 
python main.py --mode kfold --label maternalhealth_kfold_resnet_0002-0  
python main.py --mode kfold --label maternalhealth_kfold_transformer_0000-0 
python main.py --mode kfold --label maternalhealth_kfold_transformer_0001-0 
python main.py --mode kfold --label maternalhealth_kfold_transformer_0002-0 
python main.py --mode kfold_aggregate

python main.py --mode compare --label maternalhealth_kfold_mlp_0000-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_mlp_0001-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_mlp_0002-0  --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_resnet_0000-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_resnet_0001-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_resnet_0002-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_transformer_0000-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_transformer_0001-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_transformer_0002-0 --best.metric acc
python main.py --mode aggregate_compare --best.metric acc

######################################################
# Instruction Manual (DEV and DEBUG version)
######################################################

python main.py --mode preprocess_normalize
python main.py --mode preprocess_normalize
python main.py --mode kfold --label maternalhealth_kfold_mlp_0000-789 --DEV_ITER 4 --toggle 1111

# for counting param
python main.py --mode kfold --label maternalhealth_kfold_mlp_0000-789 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_mlp_0001-789 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_mlp_0002-789 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_resnet_0000-667 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_resnet_0001-667 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_resnet_0002-667 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_transformer_0000-667 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_transformer_0001-667 --toggle 0
python main.py --mode kfold --label maternalhealth_kfold_transformer_0002-667 --toggle 0