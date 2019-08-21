import pandas as pd
import numpy as np
from MDLP import *
from preprocessing import clean_default_credit
from sklearn.model_selection import train_test_split


RANDOM_STATE = 404

# Parameters
# Input file :
originalDataset = "data/default_credit/tentativePreDiscretized.csv"
# Output file :
trainingSet = "data/default_credit/default_credit_train_total.csv"
testSet = "data/default_credit/default_credit_test_total.csv"


### loading/cleaning the data
df = clean_default_credit(originalDataset)

### data type
num_attribs = []
cat_attribs = ["LIMIT_BAL", "SEX", 
"EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", 
"PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", 
"BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", 
"PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
decision = ["default_payment_next_month"]


### one-hot encoding for categorical data
y = df.default_payment_next_month.values
df.drop(labels=["default_payment_next_month"], axis = 1, inplace = True)
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

df['default_payment_next_month'] = y

#### creating the final file with all attributes binarized
y = df.default_payment_next_month.values
df.drop(labels=["default_payment_next_month"], axis = 1, inplace = True)
X = df
X = pd.get_dummies(X)

columns = list(X)
columns_final = []

columns_dict = dict(zip(columns, columns_final))

df = pd.DataFrame(X, columns=columns)

df.rename(columns=columns_dict, inplace=True)

df['default_payment_next_month'] = y

df.to_csv('data/default_credit/default_credit_binary.csv', encoding='utf-8', index=False)

## creating train/test set
df_train, df_test = train_test_split(df, test_size=0.33, stratify=df['default_payment_next_month'], random_state = RANDOM_STATE)

df_train.to_csv(trainingSet, encoding='utf-8', index=False)

df_test.to_csv(testSet, encoding='utf-8', index=False)

