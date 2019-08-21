import pandas as pd
import numpy as np
from MDLP import *
from preprocessing import clean_compas
from sklearn.model_selection import train_test_split


RANDOM_STATE = 404

# Parameters
# Input file :
originalDataset = "data/compas/compas_clean.csv"
# Output file :
trainingSet = "data/compas/compas_train_total.csv"
testSet = "data/compas/compas_test_total.csv"


### loading/cleaning the data
df = clean_compas(originalDataset)

### data type
num_attribs = ["age", "juv_misd_count", "juv_other_count", "priors_count"]
cat_attribs = ["sex", "age", "race"]
decision = ["two_year_recid"]


### one-hot encoding for categorical data
y = df.two_year_recid.values
df.drop(labels=["two_year_recid"], axis = 1, inplace = True)
X = df
X = pd.get_dummies(X)

columns = list(X)
columns_final = []

#function to search columns name from dummy name
def has_root(val, elmts):
    out = (False,None)
    for elt in elmts:
        if elt in val:
            out = (True,elt)
            break
    return out
        
### changing dummy columns name

for dump_name in columns:
    check, root = has_root(dump_name, cat_attribs)
    insert_name = ""

    if(not check):
        columns_final.append(dump_name)
    else:
        insert_name = dump_name.replace(root + "_", root + ":")
        columns_final.append(insert_name)

columns_dict = dict(zip(columns, columns_final))


df = pd.DataFrame(X, columns=columns)

df.rename(columns=columns_dict, inplace=True)

df['two_year_recid'] = y
#df.drop(labels=["juvenile-felonies_=0"], axis = 1, inplace = True)
#df.drop(labels=["juvenile-misdemeanors_=0"], axis = 1, inplace = True)
#df.drop(labels=["juvenile-crimes_=0"], axis = 1, inplace = True)
#df.drop(labels=["sex:Female"], axis = 1, inplace = True)
y2 = df.c_charge_degree_M.values
y3 = df.c_charge_degree_F.values
df.drop(labels=["c_charge_degree_M"], axis = 1, inplace = True)
df.drop(labels=["c_charge_degree_F"], axis = 1, inplace = True)

#### creating bins for numerical attributes

df = MDLP_Discretizer(dataset=df, class_label="two_year_recid", features=num_attribs).get_df()

df = df.iloc[:, :-1]


#### creating the final file with all attributes binarized
y = df.two_year_recid.values
df.drop(labels=["two_year_recid"], axis = 1, inplace = True)
X = df
X = pd.get_dummies(X)

columns = list(X)
columns_final = []

columns_dict = dict(zip(columns, columns_final))

df = pd.DataFrame(X, columns=columns)

df.rename(columns=columns_dict, inplace=True)

df['charge_degree:Misdemeanor'] = y2
df['charge_degree:Felony'] = y3
df['two_year_recid'] = y

df.to_csv('data/adult/adult_binary.csv', encoding='utf-8', index=False)

## creating train/test set
df_train, df_test = train_test_split(df, test_size=0.33, stratify=df['two_year_recid'], random_state = RANDOM_STATE)

df_train.to_csv(trainingSet, encoding='utf-8', index=False)

df_test.to_csv(testSet, encoding='utf-8', index=False)

