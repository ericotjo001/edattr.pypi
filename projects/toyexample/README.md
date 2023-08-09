# Instruction Manual 
python main.py --mode preprocess_normalize
python main.py --mode kfold --label toy_kfold_mlp_0000-0
python main.py --mode kfold --label toy_kfold_mlp_0001-0
python main.py --mode kfold --label toy_kfold_mlp_0002-0
python main.py --mode kfold --label toy_kfold_resnet_0000-0
python main.py --mode kfold --label toy_kfold_resnet_0001-0
python main.py --mode kfold --label toy_kfold_resnet_0002-0
python main.py --mode kfold --label toy_kfold_transformer_0000-0
python main.py --mode kfold --label toy_kfold_transformer_0001-0
python main.py --mode kfold --label toy_kfold_transformer_0002-0
python main.py --mode kfold_aggregate 


# Instruction Manual (DEV and DEBUG version)
python main.py --mode preprocess_normalize

python main.py --mode kfold --label toy_kfold_mlp_0000-777 --n_epochs 4 --toggle 11110

python main.py --mode kfold --label toy_kfold_mlp_0000-0 --toggle 00001

python main.py --mode kfold --label toy_kfold_mlp_0000-123 --toggle 0
python main.py --mode kfold --label toy_kfold_mlp_0001-123 --toggle 0
python main.py --mode kfold --label toy_kfold_mlp_0002-123 --toggle 0
python main.py --mode kfold --label toy_kfold_resnet_0000-123 --toggle 0
python main.py --mode kfold --label toy_kfold_resnet_0001-123 --toggle 0
python main.py --mode kfold --label toy_kfold_resnet_0002-123 --toggle 0
python main.py --mode kfold --label toy_kfold_transformer_0000-123 --toggle 0
python main.py --mode kfold --label toy_kfold_transformer_0001-123 --toggle 0
python main.py --mode kfold --label toy_kfold_transformer_0002-123 --toggle 0



python main.py --mode kfold --label toy_kfold_mlp_0000-88