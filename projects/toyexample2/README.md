# Instruction Manual 
python main.py --mode preprocess_normalize
python main.py --mode kfold --label toy2_kfold_mlp_0000-0 
python main.py --mode kfold --label toy2_kfold_mlp_0001-0 
python main.py --mode kfold --label toy2_kfold_mlp_0002-0  
python main.py --mode kfold --label toy2_kfold_resnet_0000-0 
python main.py --mode kfold --label toy2_kfold_resnet_0001-0 
python main.py --mode kfold --label toy2_kfold_resnet_0002-0 
python main.py --mode kfold --label toy2_kfold_transformer_0000-0 
python main.py --mode kfold --label toy2_kfold_transformer_0001-0 
python main.py --mode kfold --label toy2_kfold_transformer_0002-0 
python main.py --mode kfold_aggregate

## compare edattr with SHAP and LIME
python main.py --mode compare --label toy2_kfold_mlp_0000-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_mlp_0001-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_mlp_0002-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_resnet_0000-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_resnet_0001-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_resnet_0002-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_transformer_0000-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_transformer_0001-0 --best.metric f1
python main.py --mode compare --label toy2_kfold_transformer_0002-0 --best.metric f1
python main.py --mode aggregate_compare --best.metric f1

# Instruction Manual (DEV and DEBUG version)
python main.py --mode preprocess_normalize
python main.py --mode kfold --label toy2_kfold_mlp_0000-123 --n_epochs 4 
python main.py --mode kfold --label toy2_kfold_mlp_0000-123 --toggle 0
python main.py --mode kfold --label toy2_kfold_mlp_0001-124 --toggle 0
python main.py --mode kfold --label toy2_kfold_mlp_0002-125 --toggle 0
python main.py --mode kfold --label toy2_kfold_resnet_0000-999 --toggle 0 
python main.py --mode kfold --label toy2_kfold_resnet_0001-999 --toggle 0 
python main.py --mode kfold --label toy2_kfold_resnet_0002-999 --toggle 0 
python main.py --mode kfold --label toy2_kfold_transformer_0000-999 --toggle 0
python main.py --mode kfold --label toy2_kfold_transformer_0001-999 --toggle 0
python main.py --mode kfold --label toy2_kfold_transformer_0002-999 --toggle 0

python main.py --mode kfold --label toy2_kfold_mlp_0000-88