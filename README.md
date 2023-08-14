# edattr: Endorsed Attributions


## Preview
This is a preview version to an upcoming preprint.

<div align="center">
<img src="https://drive.google.com/uc?export=view&id=1D7viU_kMzK3FEXQgUcnWBdDkYmeHC406" width="577"></img>
<img src="https://drive.google.com/uc?export=view&id=17FjxxIdLtvCyCPrWBPx-Hkvu-aow8Vfc" width="577"></img>
</div>

The results can be obtained with simple condensed commands like:
```
python main.py --mode preprocess_normalize

python main.py --mode kfold --label maternalhealth_kfold_mlp_0000-0 
python main.py --mode kfold --label maternalhealth_kfold_resnet_0000-0 
python main.py --mode kfold --label maternalhealth_kfold_transformer_0000-0 
python main.py --mode kfold_aggregate

python main.py --mode compare --label maternalhealth_kfold_mlp_0000-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_resnet_0000-0 --best.metric acc
python main.py --mode compare --label maternalhealth_kfold_transformer_0000-0 --best.metric acc
python main.py --mode aggregate_compare
```


# About

v0.1.0-0

This project is about endorsed attribution (edattr), essentially a voting mechanism for feature attributions. Given n feature attribution methods, we find the top m features endorsed by each method. The total votes that each feature gets is our new attribution. For example, if SHAP's top features are [x1,x2] and LIME's top features are [x1,x3], then edattr={ "x1":2, "x2":1, "x3":1 }.

Furthermore, data can be grouped or "partitioned" according to their eddatr. These partitions are then processed to yield smaller datasets, called *EEC data subsets*, which can be used for a more efficient training process.

The related paper: TBU

# FOR LOCAL SETUP 
## Getting Started

### Installations
All required packages are listed in the yml file except pytorch (which you may need to manually select based on your machine, see its official installation instruction). We recommend installing conda environment with the yml file.


**PYTORCH INSTALLATION**
For any pytorch related stuff, we install after all the standard packages (just in case)
```
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
pip3 install captum==0.6.0
```
At the time of installation, our pytorch version is 2.1.0.dev20230406+cu117

**Install standard packages**
```
cuda version 11.7
conda env update --file env.yml
```

## Data
See README.md file in projects folder to understand the types of format currently supported by our package.

====== project: fetalhealthclassifier =======
fetal_health.csv 0.2 MB
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

Using features from Cardiotocograms (CTGs) to predict mortality and morbidity (3-class)
only 2000+ records


====== project: maternalhealth =======
Maternal Health Risk Data Set.csv 30kb
https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data

6 basic features to predict maternal health status

====== project: Body signal of smoking =======
smoking.csv: 6.08MB
https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking

data shape : (55692, 27)
to detect whether this patient is smoking (binary)

====== project: heart disease =======
heart_2020_cleaned.csv 24MB
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

400k patients!
HeartDisease" as a binary ("Yes" - respondent had heart disease; "No" - respondent had no heart disease).

# To-do
1. Create Token only framework? (We have Num and Token+Num now) 
2. Perhaps deprecate k-fold framework, it feels a bit cumbersome. Maybe using repeating the standard train/val/test pipeline 5x with proper adjustment will substitue k-fold.

Erico: hi Feng, I've created \_misc/discussion.txt thread for us to discuss.



