# Instruction Manual 
python main.py --mode preprocess_normalize
python main.py --mode kfold --label fhc_kfold_mlp_0000-0 
python main.py --mode kfold --label fhc_kfold_mlp_0001-0  
python main.py --mode kfold --label fhc_kfold_mlp_0002-0 
python main.py --mode kfold --label fhc_kfold_resnet_0000-0 
python main.py --mode kfold --label fhc_kfold_resnet_0001-0 
python main.py --mode kfold --label fhc_kfold_resnet_0002-0  
python main.py --mode kfold --label fhc_kfold_transformer_0000-0 
python main.py --mode kfold --label fhc_kfold_transformer_0001-0 
python main.py --mode kfold --label fhc_kfold_transformer_0002-0 
python main.py --mode kfold_aggregate 

# Instruction Manual (DEV and DEBUG version)
python main.py --mode preprocess_normalize
python main.py --mode kfold --label fhc_kfold_mlp_0000-578 --n_epochs 2 --toggle 1111

python main.py --mode kfold --label fhc_kfold_mlp_0000-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_mlp_0001-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_mlp_0002-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_resnet_0000-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_resnet_0001-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_resnet_0002-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_transformer_0000-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_transformer_0001-578 --toggle 0
python main.py --mode kfold --label fhc_kfold_transformer_0002-578 --toggle 0

