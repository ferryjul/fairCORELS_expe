import pandas as pd
import numpy as np
from MDLP import *
from preprocessing import clean_german_credit
from sklearn.model_selection import train_test_split


RANDOM_STATE = 404

# Parameters
# Input file :
originalDataset = "data/german_credit/german_credit_dataset.csv"
# Output file :
trainingSet = "data/german_credit/german_credit_train_total.csv"
testSet = "data/german_credit/german_credit_test_total.csv"


### loading/cleaning the data
df = clean_german_credit(originalDataset)

### data type
num_attribs = ["credit_amount", "age", "credit_duration_months"]
cat_attribs = ["account_balance", "previous_credit_payment_status", 
"credit_purpose", "savings", "employment_duration", "installment_rate(perc_disp_income)", 
"marital_status", "guarantor", "residence_duration", "current_assets", "other_credits", 
"apartment_type", "bank_credits", "occupation", "dependents", "telephone_yes", "foreign_worker_yes", "telephone_no", "foreign_worker_no"]
decision = ["credit_rating"]


### one-hot encoding for categorical data
y = df.credit_rating.values
df.drop(labels=["credit_rating"], axis = 1, inplace = True)
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

df['credit_rating'] = y
#df.drop(labels=["juvenile-felonies_=0"], axis = 1, inplace = True)
#df.drop(labels=["juvenile-misdemeanors_=0"], axis = 1, inplace = True)
#df.drop(labels=["juvenile-crimes_=0"], axis = 1, inplace = True)
#df.drop(labels=["sex:Female"], axis = 1, inplace = True)
'''y2 = df.c_charge_degree_M.values
y3 = df.c_charge_degree_F.values
df.drop(labels=["c_charge_degree_M"], axis = 1, inplace = True)
df.drop(labels=["c_charge_degree_F"], axis = 1, inplace = True)'''

#### creating bins for numerical attributes

df = MDLP_Discretizer(dataset=df, class_label="credit_rating", features=num_attribs).get_df()

df = df.iloc[:, :-1]


#### creating the final file with all attributes binarized
y = df.credit_rating.values
df.drop(labels=["credit_rating"], axis = 1, inplace = True)
X = df
X = pd.get_dummies(X)

columns = list(X)
columns_final = []

columns_dict = dict(zip(columns, columns_final))

df = pd.DataFrame(X, columns=columns)

df.rename(columns=columns_dict, inplace=True)

'''df['charge_degree:Misdemeanor'] = y2
df['charge_degree:Felony'] = y3'''
df['credit_rating'] = y

df.to_csv('data/german_credit/german_credit_binary.csv', encoding='utf-8', index=False)

## creating train/test set
df_train, df_test = train_test_split(df, test_size=0.33, stratify=df['credit_rating'], random_state = RANDOM_STATE)

df_train.to_csv(trainingSet, encoding='utf-8', index=False)

df_test.to_csv(testSet, encoding='utf-8', index=False)

