import pandas as pd
import numpy as np
from MDLP import *
from preprocessing import clean_adult
from sklearn.model_selection import train_test_split


RANDOM_STATE = 404



### loading/cleaning the data
df = clean_adult('data/adult/adult.csv')

### data type
num_attribs = ["age", "hours_per_week", "capital_gain", "capital_loss"]
cat_attribs = ["workclass", "education", "marital_status", "occupation", "gender"]
decision = ["income"]


### one-hot encoding for categorical data
y = df.income.values
df.drop(labels=["income"], axis = 1, inplace = True)
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

df['income'] = y

#### creating bins for numerical attributes

df = MDLP_Discretizer(dataset=df, class_label="income", features=num_attribs).get_df()

df = df.iloc[:, :-1]


#### creating the final file with all attributes binarized
y = df.income.values
df.drop(labels=["income"], axis = 1, inplace = True)
X = df
X = pd.get_dummies(X)

columns = list(X)
columns_final = []

### changing dummy columns name

for dump_name in columns:
    check, root = has_root(dump_name, num_attribs)
    insert_name = ""

    if(not check):
        columns_final.append(dump_name)
    else:
        insert_name = dump_name.replace(root + "_", root + ":")
        columns_final.append(insert_name)


columns_dict = dict(zip(columns, columns_final))

df = pd.DataFrame(X, columns=columns)

df.rename(columns=columns_dict, inplace=True)

df['income'] = y

df.to_csv('data/adult/adult_binary.csv', encoding='utf-8', index=False)

## creating train/test set
df_train, df_test = train_test_split(df, test_size=0.33, stratify=df['income'], random_state = RANDOM_STATE)

df_train.to_csv('data/adult/adult_train_binary.csv', encoding='utf-8', index=False)

df_test.to_csv('data/adult/adult_test_binary.csv', encoding='utf-8', index=False)

