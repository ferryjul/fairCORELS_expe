"""
First we import modules for model building and data
processing.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib  
import codecs
import matplotlib.pyplot as plt
import os
from utils import *

"""
Now, we import the two key methods from fairml.
audit_model takes:

- (required) black-box function, which is the model to be audited
- (required) sample_data to be perturbed for querying the function. This has to be a pandas dataframe with no missing data.

- other optional parameters that control the mechanics of the auditing process, for example:
  - number_of_runs : number of iterations to perform
  - interactions : flag to enable checking model dependence on interactions.

audit_model returns an overloaded dictionary where keys are the column names of input pandas dataframe and values are lists containing model  dependence on that particular feature. These lists of size number_of_runs.

"""
from fairml import audit_model
from fairml import plot_dependencies

def cleanFeaturesDict(aDict, k): # Only keeps the k most significant features of dict aDict and returns the new dict
    # Loop not opt (O(k*len(aDict))), but in my script it is called only once so OK
    res = aDict
    while(len(res) > k):
        currMin = np.inf
        currMinKey = ""
        for f in res:
            if(abs(res.get(f)) < currMin):
                currMin = abs(res.get(f))
                currMinKey = f
        res.pop(currMinKey)
    return res

def correctSign(aDictProxy, aDictNoProxy): # Only keeps the k most significant features of dict aDict and returns the new dict
    # Loop not opt (O(k*len(aDict))), but in my script it is called only once so OK
    res = aDictNoProxy
    for f in aDictProxy:
        if (aDictProxy.get(f) < 0 and res.get(f) > 0) or (aDictProxy.get(f) > 0 and res.get(f) < 0):
            el = res.pop(f)
            el = (-1)*el
            res.update({f:el})        
    return res

def computeDiff(totalRaw, totalBis):
    # First we build a dict with same scale
    maxProxy = 0
    maxBis = 0
    for feat in totalRaw:
        proxyVal = totalRaw.get(feat)
        selfCompVal = totalBis.get(feat)
        if maxProxy < proxyVal :
            maxProxy = proxyVal
        if maxBis < selfCompVal :
            maxBis = selfCompVal
    rangeFact = maxBis/maxProxy
    total = {}
    for feat in totalRaw:
        newVal = totalRaw.get(feat)*rangeFact 
        total.update({feat:newVal})
    totalDiff = 0
    NB = 0
    for feat in total:
        proxyVal = total.get(feat)
        selfCompVal = totalBis.get(feat)
        if selfCompVal == 0:
            fact = abs(proxyVal)          
        else:
            if proxyVal == 0:
                fact = selfCompVal
            else:
                fact = (abs(proxyVal-selfCompVal)/abs(selfCompVal))
        print("feat = ", feat, "fact = ", fact, "proxy val = ", proxyVal, ", selfCompVal = ", selfCompVal)
        if proxyVal > 0.01 or selfCompVal > 0.01: #we ignore relatively not important features
            totalDiff = totalDiff + fact
            NB = NB + 1
    return (totalDiff/NB) # Average relative difference
    

dataset_name = "german_credit" #Choose here the dataset to be audited. Corresponding model will be found in gen_files/dataset_name/corels_raw.txt
nbFeat = 25 # Choose here the maximum number of features that you want to appear in the feature dependence plot
# --- Input files : -----------------------
itemSetFile = "./data/adult/adult_itemset.txt"
itemSetNamesFile =  "./data/adult/adult_itemset_name.txt"
fullFeatures = "./data/adult/adult_full.feature"
fullLabels = "./data/adult/adult_full.label"
trainMinor = ""
posPrediction = ">50K"
analysisSet = "./corels_datasets/data/adult/adult_full_binary.csv"
if dataset_name == "compas":
    analysisSet = "./corels_datasets/data/compas/compas_full_binary.csv"
    posPrediction = "two_year_recid"
    itemSetFile = "./data/compas/compas_itemset.txt"
    itemSetNamesFile =  "./data/compas/compas_itemset_name.txt"
    fullFeatures = "./data/compas/compas_full.feature"
    fullLabels = "./data/compas/compas_full.label"
    #trainMinor = "./data/compas/compas_train.minor"
    trainMinor = ""
    #testFeatures = "./data/compas/compas_test.out"
    #testLabels = "./data/compas/compas_test.label"
    #testMinor = "./data/compas/processed/_test.minor"
elif dataset_name == "german_credit":
    posPrediction = "credit_rating"
    analysisSet = "./corels_datasets/data/german_credit/german_credit_full_binary.csv"
    itemSetFile = "./data/german_credit/german_credit_itemset.txt"
    itemSetNamesFile =  "./data/german_credit/german_credit_itemset_name.txt"
    fullFeatures = "./data/german_credit/german_credit_full.feature"
    fullLabels = "./data/german_credit/german_credit_full.label"
    trainMinor = ""
elif dataset_name == "default_credit":
    posPrediction = "default_payment_next_month"
    dataset_name = "default_credit"
    analysisSet = "./corels_datasets/data/default_credit/default_credit_binary_full.csv"
    itemSetFile = "./data/default_credit/default_credit_itemset.txt"
    itemSetNamesFile =  "./data/default_credit/default_credit_itemset_name.txt"
    fullFeatures = "./data/%s/default_credit_full.feature" %dataset_name
    fullLabels = "./data/%s/default_credit_full.label" %dataset_name

CORELS_raw_file = "./gen_files/%s/corels_raw.txt" %dataset_name
modelFile = "gen_files/%s/model_dumped.mdl" %dataset_name
print("Récupération, formattage et sauvegarde du modèle correspondant")
model = CORELS()
model.treatCORELSraw(result_file=CORELS_raw_file, itemset_file=itemSetFile, itemset_name_file=itemSetNamesFile)
model.posPrediction = posPrediction
joblib.dump(model, modelFile, compress=9)

#testSet = "data/adult/adult_full.feature"
# Read model from file
adult_mdl       = joblib.load(modelFile)

print("Rule list to be analyzed : \n", make_string(adult_mdl))
# Read binarized dataset from csv
adult_data = pd.read_csv(
    filepath_or_buffer=analysisSet)

# Attach original dataframe to model
adult_mdl.attachFrame(adult_data)

# create feature and design matrix for model building.
if(dataset_name == "adult"):
    adult_rating = adult_data.income.values
    adult_data = adult_data.drop("income", 1)
elif dataset_name == "compas":
    adult_rating = adult_data.two_year_recid.values
    adult_data = adult_data.drop("two_year_recid", 1)
elif dataset_name == "default_credit":
    adult_rating = adult_data.default_payment_next_month.values
    adult_data = adult_data.drop("default_payment_next_month", 1)
else:
    adult_rating = adult_data.credit_rating.values
    adult_data = adult_data.drop("credit_rating", 1)

#adult_mdl.predict(adult_data.values)
gen_both = True
# we fit a quick and dirty logistic regression sklearn
# model here.
if gen_both :
    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(adult_data.values, adult_mdl.predict(adult_data.values))#adult_mdl.predict(adult_data.values))
    # print(clf.predict(adult_data))
    #  call audit model with model
    #clf.predict
    total, _ = audit_model(clf.predict, adult_data)
    totalbis, _ = audit_model(adult_mdl.predict, adult_data)
    # print feature importance
    #print(total)

    # get corresponding dictionnary
    featuresDict = total.median()
    featuresFictBis = totalbis.median()
    #print(featuresDict)
    featuresFictBis = correctSign(featuresDict, featuresFictBis)

    # should be normalized
    diff = computeDiff(featuresDict, featuresFictBis)

    featuresDict = cleanFeaturesDict(featuresDict, nbFeat)
    featuresFictBis = cleanFeaturesDict(featuresFictBis, nbFeat)
    print("Average relative difference between direct audit and proxy audit valuations : ", 100*diff, " %")

    # generate feature dependence plot
    fig = plot_dependencies(
        featuresDict,
        reverse_values=False,
        title="FairML feature dependence (with proxy)",
        fig_size=(6, 9)
    )
    plt.savefig("./plots/fairml_ldp_proxy.eps", transparent=False, bbox_inches='tight')

    figBis = plot_dependencies(
        featuresFictBis,
        reverse_values=False,
        title="FairML feature dependence (without proxy)",
        fig_size=(6, 9)
    )
    plt.savefig("./plots/fairml_ldp_no_proxy.eps", transparent=False, bbox_inches='tight')
else:
    clf = LogisticRegression(penalty='l2', C=0.01)
    clf.fit(adult_data.values, adult_rating)
    total, _ = audit_model(clf.predict, adult_data)

    featuresDict = total.median()

    featuresDict = cleanFeaturesDict(featuresDict, nbFeat)
    
    # generate feature dependence plot
    fig = plot_dependencies(
        featuresDict,
        reverse_values=False,
        title="FairML feature dependence (with proxy)",
        fig_size=(6, 9)
    )
    plt.savefig("./plots/fairml_ldp_proxy.eps", transparent=False, bbox_inches='tight')