###################################################
# Instruction Manual 
###################################################
# run preprocessing twice. The first to set TOKEN_FEATURES and NUMERICAL_FEATURES
python main.py --mode preprocess_normalize
python main.py --mode preprocess_normalize

python main.py --mode standard --label heartdisease_standard_mlp_0000-0
python main.py --mode standard --label heartdisease_standard_mlp_0001-0
python main.py --mode standard --label heartdisease_standard_mlp_0002-0
python main.py --mode standard --label heartdisease_standard_resnet_0000-0
python main.py --mode standard --label heartdisease_standard_resnet_0001-0
python main.py --mode standard --label heartdisease_standard_resnet_0002-0
python main.py --mode standard --label heartdisease_standard_transformer_0000-0
python main.py --mode standard --label heartdisease_standard_transformer_0001-0
python main.py --mode standard --label heartdisease_standard_transformer_0002-0
python main.py --mode standard_aggregate

python main.py --mode compare --label heartdisease_standard_mlp_0000-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_mlp_0001-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_mlp_0002-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_resnet_0000-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_resnet_0001-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_resnet_0002-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_transformer_0000-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_transformer_0001-0 --best.metric acc
python main.py --mode compare --label heartdisease_standard_transformer_0002-0 --best.metric acc
python main.py --mode aggregate_compare --best.metric acc

###################################################
# Instruction Manual (DEV and DEBUG version) 
###################################################
python main.py --mode preprocess_normalize
python main.py --mode preprocess_normalize

python main.py --mode standard --label heartdisease_standard_mlp_0000-123 --toggle 1111
python main.py --mode standard --label heartdisease_standard_mlp_0001-124 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111
python main.py --mode standard --label heartdisease_standard_mlp_0002-125 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111

python main.py --mode standard --label heartdisease_standard_resnet_0000-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17 --toggle 1111
python main.py --mode standard --label heartdisease_standard_resnet_0001-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111
python main.py --mode standard --label heartdisease_standard_resnet_0002-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111

python main.py --mode standard --label heartdisease_standard_transformer_0000-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17 --toggle 1111
python main.py --mode standard --label heartdisease_standard_transformer_0001-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111
python main.py --mode standard --label heartdisease_standard_transformer_0002-999 --n_epochs 2 --eec_n_epochs 7 --DEV_ITER 17  --toggle 1111



# param counts
python main.py --mode standard --label heartdisease_standard_mlp_0000-123 --toggle 0
python main.py --mode standard --label heartdisease_standard_mlp_0001-124 --toggle 0
python main.py --mode standard --label heartdisease_standard_mlp_0002-125 --toggle 0
python main.py --mode standard --label heartdisease_standard_resnet_0000-999 --toggle 0
python main.py --mode standard --label heartdisease_standard_resnet_0001-999 --toggle 0
python main.py --mode standard --label heartdisease_standard_resnet_0002-999 --toggle 0
python main.py --mode standard --label heartdisease_standard_transformer_0000-999 --toggle 0
python main.py --mode standard --label heartdisease_standard_transformer_0001-999 --toggle 0
python main.py --mode standard --label heartdisease_standard_transformer_0002-999 --toggle 0
