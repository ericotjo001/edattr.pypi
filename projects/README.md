# About
This "projects" folder contains applications of endorsed attribution (edattr) pipeline on different format of datasets.

# Different Types of Projects
Here we group our available projects according to their data types.

**1. Numerical + Tokens**. Data consists of (1) a column of target variable, (2) columns of numbers (NUMERICAL, floating point number, can be ordered like real numbers) and (3) other columns with words or short phrases (TOKEN, like 'Y','N', '10-20',"20-30", ">30", "Male", "Female"). We can specify each of them manually. Example projects: 
- standard train/val/test: toyexample3, smoking, heartdisease. 
- kfold: toyexample2, maternalhealth

**2. Numerical**. Data consists of (1) a column of target variable (2) columns of numbers (NUMERICAL) only. Example projects:
- kfold: toyexample, fetalhealthclassifier


**Notes**
1. here, what we mean by *tokens* is anything that can be treated like a word in a sentence (we adopt this from the common way NLP researchers/engineers process their text data). For example, if we have a column of "age" with entries like "10-20", "20-30", ">30", then we can treat each of them as token 0,1,2, each token also being represented by a high dimensional vector. We intend to tokenize everything that is converted into one-hot vectors or ordinal numbers in traditional machine learning settings.
2. So far, we use k-fold for smaller data and standard pipeline (1x train, 1x val and 2x test) for larger data 


# gephi
To create graph visual representation in endorsement.visual train/val/test/all, use Gephi.

ALL GRAPHS UNDIRECTED.
Node levels set as float (no. of endorsements)
Edge merge strategy: Average

## Configurations
Node colors: Partition by y0    
    0: A6D98F (green)
    1: E3BF5F (orange)
    2: FF7873 (redder)
Node size: 
    ranking by degree | size in [7,57]
Node Label color: 
    partition by level 
    1: 025203  (dark green, endo=1)
    2: 03AB06  (light green, endo=2)
Node Label size
    ranking by level | 
    min/max size is [1.2,2]
    level: 1,2 refers to no. of endorsements

NOTE: Node size must be set to "scaled" (see bottom bar)

Edge colors: Partition by y0    
    Same as Node colors

Arial Bold 24, bar at half

Tips: use Force Atlas then Fruchterman Reingold to rerrange layout

# Debug run  
The following runs through some projects once, just make sure that they complete (after all normalize preprocessing)

```
cd toyexample
python main.py --mode preprocess_normalize
python main.py --mode kfold --label toy_kfold_mlp_0000-123 --toggle 1111
cd ../
cd toyexample2
python main.py --mode preprocess_normalize
python main.py --mode kfold --label toy2_kfold_mlp_0000-123 --toggle 1111
cd ../
cd toyexample3
python main.py --mode preprocess_normalize
python main.py --mode standard --label toy3_standard_mlp_0000-123 --toggle 1111
cd ../
cd fetalhealthclassifier
python main.py --mode preprocess_normalize
python main.py --mode kfold --label fhc_kfold_mlp_0000-578 --n_epochs 8 --toggle 1111
cd ../
cd medinsurance
python main.py --mode preprocess_normalize
python main.py --mode kfold --label medinsurance_kfold_mlp_0000-789 --n_epochs 8 --toggle 1111
```



