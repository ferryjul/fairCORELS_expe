import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from joblib import Parallel, delayed         
import codecs
import sys
import math
#from collections import namedtuple
import matplotlib.pyplot as plt
import csv
import heapq

def computeCORELSCross(_mode, _h, kFold, lambdaFc, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict):
    totList = Parallel(n_jobs=-1)(delayed(calc)(i = kNb, mode = _mode, h = _h, l = lambdaFc, dataset_name=dataset_name, maxNBnodes=maxNBnodes, fairnessProtectedFeature=fairnessProtectedFeature, fairnessUnprotectedFeature=fairnessUnprotectedFeature, beta=beta, unprotArg=unprotArg, trainMinor=trainMinor, itemSetFile=itemSetFile, itemSetNamesFile=itemSetNamesFile,kClosestDict=kClosestDict) for kNb in range(kFold))
    accListTest = []
    fairListTest = []
    accListTrain = []
    fairListTrain = []
    modelsList = []
    inconsList = []
    for i in range(len(totList)):
        accListTest.append(totList[i][0])
        fairListTest.append(totList[i][1])
        accListTrain.append(totList[i][2])
        fairListTrain.append(totList[i][3])
        modelsList.append(totList[i][4])
        inconsList.append(totList[i][5])
    #print("----------------------------------")
    #print("av acc train => ", accListTrain, "\n")
    #print("av acc test => ", accListTest, "\n")
    avAccTest = average(accListTest)
    avFairTest = average(fairListTest)
    avAccTrain = average(accListTrain)
    avFairTrain = average(fairListTrain)
    avInconsist = average(inconsList)
    stdAcc = computeSTD(accListTest)
    stdFair = computeSTD(fairListTest)
    #print("Best model is %d\n" %findBest(avAccTest, avFairTest, accListTest, fairListTest))
    bestIndex = findBest(accListTest, fairListTest)
    bestModel = modelsList[bestIndex]
    return [avAccTest, avFairTest, avAccTrain, avFairTrain, stdAcc, stdFair, bestModel, avInconsist]

def kNN(dataMatrix, k, kF): # returns dictionnary {instance#:[kCloserInstances#]}
    totalInstances = len(dataMatrix[0])-1
    res = {}
    print("Generating ", k, "closest-neighbours for ", totalInstances, " instances.")
    seuil = 0.1
    for inst in range(1,totalInstances+1): # We loop on each individual in the set
        # Init the data structure
        heap = []
        for inst2 in range(1,totalInstances+1):
            totalDist = 0
            for feat in range(len(dataMatrix)): # We loop though attributes
                totalDist = totalDist + abs(int(dataMatrix[feat][inst])-int(dataMatrix[feat][inst2]))
            totalDist = (-1)*totalDist # so we can order the heap by min to actually access max
            if len(heap) < k:
                heapq.heappush(heap, (totalDist,inst2))
            elif heap[0] < (totalDist,inst2):
                heapq.heappop(heap)
                heapq.heappush(heap, (totalDist,inst2))
        res.update({inst:heap})
        if inst/totalInstances >= seuil:
            print("[FOLD ", kF, "] ", 100*seuil, "/100 done.")
            seuil = seuil + 0.1
        #print(k, " closer elements from ", inst, " : ", heap)
    return res

def computekNN(kF, k):
    testFeatures = "_tmp/temp_test_features_%d.feature" % (kF)
    testLabels = "_tmp/temp_test_labels_%d.label" % (kF)
    matrix = build_matrix(testFeatures, testLabels)
    return kNN(matrix,k, kF)

def clean_lists_with_lambda(s1, s2, s3, s4, s5, s6, s7): # Removes repetitions and dominated solutions
    s1Tmp = []
    s2Tmp = []
    s3Tmp = []
    s4Tmp = []
    s5Tmp = []
    s6Tmp = []
    s7Tmp = []
    for i in range(len(s1)):
        isDominated = False
        repetition = False
        for j in range(len(s1)):
            if i != j:
                if ((s1[i]<s1[j]) and (s2[i]<=s2[j])) or ((s2[i]<s2[j]) and (s1[i]<=s1[j])):
                    isDominated = True
                    print("[%lf,%lf] dominated by [%lf,%lf]\n" %(s1[i], s2[i], s1[j], s2[j]))
                elif (s1[i] == s1[j]) and (s2[i] == s2[j] and j < i):
                    repetition = True
        if(not isDominated and not repetition):
            s1Tmp.append(s1[i])
            s2Tmp.append(s2[i])
            s3Tmp.append(s3[i])
            s4Tmp.append(s4[i])
            s5Tmp.append(s5[i])
            s6Tmp.append(s6[i])
            s7Tmp.append(s7[i])
    return s1Tmp, s2Tmp, s3Tmp, s4Tmp, s5Tmp, s6Tmp, s7Tmp

def clean_lists_no_lambda(s1, s2, s3, s4, s5, s6): # Removes repetitions and dominated solutions
    s1Tmp = []
    s2Tmp = []
    s3Tmp = []
    s4Tmp = []
    s5Tmp = []
    s6Tmp = []
    for i in range(len(s1)):
        isDominated = False
        repetition = False
        for j in range(len(s1)):
            if i != j:
                if ((s1[i]<s1[j]) and (s2[i]<=s2[j])) or ((s2[i]<s2[j]) and (s1[i]<=s1[j])):
                    isDominated = True
                    print("[%lf,%lf] dominated by [%lf,%lf]\n" %(s1[i], s2[i], s1[j], s2[j]))
                elif (s1[i] == s1[j]) and (s2[i] == s2[j] and j < i):
                    repetition = True
        if(not isDominated and not repetition):
            s1Tmp.append(s1[i])
            s2Tmp.append(s2[i])
            s3Tmp.append(s3[i])
            s4Tmp.append(s4[i])
            s5Tmp.append(s5[i])
            s6Tmp.append(s6[i])
    return s1Tmp, s2Tmp, s3Tmp, s4Tmp, s5Tmp, s6Tmp

def make_string(m):
    tempModel = ""
    pred_list = m.pred_description_
    for i in range(len(pred_list)): 
        pred = pred_list[i][1:-1]
        if (i == 0):               
            try:
                rule = m.rule_description_[i].replace('{', '').replace('}', '').split(',')
            except:
                rule = '1'
            tempModel = tempModel + "IF "
            for j, r in enumerate(rule):
                tempModel = tempModel + r
                if j < len(rule) - 1:
                    tempModel = tempModel + " AND "
            # Ecriture décision correspondante
            tempModel = tempModel + " THEN "
            tempModel = tempModel + pred
            tempModel = tempModel + "\n"
        elif i == len(m.pred_description_) - 1:
            # Ecriture décision par défaut
            tempModel = tempModel + "ELSE "
            tempModel = tempModel + pred
            tempModel = tempModel + "\n"
        else:
            rule = m.rule_description_[i].replace('{', '').replace('}', '').split(',')
            # Ecriture condition
            tempModel = tempModel + "ELSE IF "
            for j, r in enumerate(rule):
                tempModel = tempModel + r
                if j < len(rule) - 1:
                   tempModel = tempModel + " AND "
            # Ecriture décision correspondante
            tempModel = tempModel + " THEN "
            tempModel = tempModel + pred
            tempModel = tempModel + "\n"
    return tempModel

def average(aList):
    tot = 0.0
    for k in aList:
        tot = tot + k
    return tot/len(aList)

def findBest(valList1, valList2):
    bigger = -1 # Small number
    bestIndex = 0
    for i in range(len(valList1)):
        currDelta = valList1[i] - (1-valList2[i])
        #print("currDelta = %lf, bigger = %lf\n" %(currDelta, bigger))
        if currDelta > bigger:
            bigger = currDelta
            bestIndex = i
    return bestIndex

def pred(model, inst, dataMatrix, featuresDict):
    pred_list = model.pred_description_
    for i in range(len(pred_list)): 
        if i == len(model.pred_description_) - 1: # if default decision
            pred = pred_list[i][1:-1]
            if pred==dataMatrix[len(dataMatrix)-1][0]:
                return 1
            else:
                return 0
        else:
            pred = pred_list[i][1:-1]
            if pred==dataMatrix[len(dataMatrix)-1][0]:
                pred = 1
            else:
                pred = 0  
            ruleL = model.rule_description_[i].replace('{', '').replace('}', '').split(',')
            correct = 1
            for i in range(len(ruleL)):
                if int(dataMatrix[featuresDict.get(ruleL[i])][inst]) == 0:
                    correct = 0
            if correct == 1: # all conditions matched
                return pred

def computeInconsistency(neighboursList, dataMatrix, model, featuresDict):
    #print("Computing inconsistency")
    values = []
    #print(neighboursList)
    for inst in neighboursList:
        #print("inst =", inst)
        neighbours = neighboursList.get(inst)
        totPreds = 0
        for nn in neighbours:
            n = nn[1]
            totPreds = totPreds + pred(model,n, dataMatrix, featuresDict)
        neighboursAverage = totPreds/len(neighbours)
        instPred = pred(model,inst, dataMatrix, featuresDict)
        values.append(abs(instPred - neighboursAverage))
    av = average(values)
    #print("inconsistency = ", av)
    return av

def calc(mode, h, i, l, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict):
    trainFeatures = "_tmp/temp_train_features_%d.feature" % (i)
    testFeatures = "_tmp/temp_test_features_%d.feature" % (i)
    trainLabels = "_tmp/temp_train_labels_%d.label" % (i)
    testLabels = "_tmp/temp_test_labels_%d.label" % (i)
    CORELS_raw_file = "./gen_files/%s/corels_raw_%d.txt" %(dataset_name, i)
    # Lancement de CORELS
    com = './algs/corels/src/corels -n %d -r %f -x %f -m %d -z %f -i 1 -p 1 -b %s %s %s %s %s> %s' % (maxNBnodes, l, fairnessProtectedFeature, mode, beta, h, unprotArg, trainFeatures, trainLabels, trainMinor, CORELS_raw_file)
    #if i == 0:
    #    print("[PREMIER THREAD] Lancement de CORELS avec la commande : ", com)
    os.system(com)

    # Récupération et analyse du fichier généréitemsets, data_name, label_name = enumerator.load_itemsets_and_names(itemset_file, itemset_name_file) par CORELS
    model = CORELS()
    model.treatCORELSraw(result_file=CORELS_raw_file, itemset_file=itemSetFile, itemset_name_file=itemSetNamesFile)

    # Récupération modèles depuis fichier dumped et exportation dans fichier .txt
    adult_mdl       = model
    # Préparation de la matrice et du dictionnaire
    trainMatrix = build_matrix(trainFeatures, trainLabels)
    testMatrix = build_matrix(testFeatures, testLabels)
    featuresDict = build_dictionnary(trainFeatures) # Train ou test, peu importe ici vu qu'on ne regarde que les "headers"
    # Calcul des métriques
    training_acc = getAcc(featuresDict, trainMatrix, adult_mdl)
    training_fair = getFair(featuresDict, trainMatrix, adult_mdl, fairnessProtectedFeature-1, max([-1,fairnessUnprotectedFeature-1]))
    test_acc = getAcc(featuresDict, testMatrix, adult_mdl)
    test_fair = getFair(featuresDict, testMatrix, adult_mdl, fairnessProtectedFeature-1, max([-1,fairnessUnprotectedFeature-1]))
    # We check that the metrics computed are done the same way than in CORELS :
    if mode == 3:
        if training_fair < float(h.split(' ')[1]):
            print("[WARNING] Unsufficient result from CORELS ; expected training was %lf, got %lf\n Command was : %s" %(float(h.split(' ')[1]), training_fair, com))
    if (abs(training_acc - adult_mdl.finalAcc_) > 0.000001) or (abs(training_fair - adult_mdl.finalFair_) > 0.000001):
        print("\n[WARNING] Python computed : training acc = %lf, training fair = %lf\nCORELS computed : training acc = %lf, training fair = %lf\nCommand was : %s\n" %(training_acc, training_fair, adult_mdl.finalAcc_, adult_mdl.finalFair_, com))
    #Calcul de l'inconsistency :
    inconsist = computeInconsistency(kClosestDict[i], testMatrix, adult_mdl, featuresDict)
    return [test_acc, test_fair, training_acc, training_fair, make_string(adult_mdl), inconsist]



def build_matrix(featuresSet, labelsSet):
    # The matrix is actually a list of lists
    # Each list corresponds to a feature (or the label for the last one)
    # Each element of a list corresponds to an individual
    # For instance, matrix[0][4] has value of feature 0 for individual 4
    matrix = []
    with codecs.open(featuresSet, 'r', 'utf-8', 'ignore') as f:
        for line in f:
            featureValues = line.replace('{', '').replace('}', '').replace('\n', '').split(' ')
            #print(featureValues[0])
            matrix.append(featureValues)
    with codecs.open(labelsSet, 'r', 'utf-8', 'ignore') as f:
       for line in f:
            featureValues = line.replace('{', '').replace('}', '').replace('\n', '').split(' ')
            #print(featureValues[0])
            matrix.append(featureValues)
    return matrix

def build_dictionnary(featuresSet):
    featuresDict = {}
    with codecs.open(featuresSet, 'r', 'utf-8', 'ignore') as f:
        i = 0
        for line in f:
            featureName = line.replace('{', '').replace('}', '').replace('\n', '').split(' ')[0]
            featuresDict.update( {featureName : i} )
            #print(featureName)
            i = i + 1
    return featuresDict
    
def load_itemsets_and_names(itemset_file, itemset_name_file):
    itemsets = []
    data_name = []
    label_name = []
    with open(itemset_file, 'r') as f:
        for line in f:
            itemset = set([int(r) for r in line.split()])
            itemsets.append(itemset)
        f.close()
    with open(itemset_name_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            elif i == 1:
                label_name.append(line.strip())
            elif i == 2:
                continue
            else:
                data_name.append(line.strip())
        f.close()
    return itemsets, data_name, label_name

def convert_file(original_data_file, original_label_file, new_data_file, new_itemset_file=None, new_itemset_name_file=None):
    records = []
    record_names = []
    labels = []
    label_names = []
    with open(original_data_file, 'r') as f:
        for line in f:
            record = line.split()
            records.append([int(r) for r in record[1:]])
            record_names.append(record[0][1:-1])
        f.close()
    with open(original_label_file, 'r') as f:
        for line in f:
            record = line.split()
            labels.append([int(r) for r in record[1:]])
            label_names.append(record[0][1:-1])
        f.close()
    unique_record_names = [name for name in record_names if len(name.split(','))==1]
    unique_record_index = [i for i, name in enumerate(record_names) if len(name.split(','))==1]
    records = np.array(records).T
    records = records[:, unique_record_index]
    labels = np.array(labels).T
    with open(new_data_file, 'w') as f:
        for i in range(records.shape[0]):
            idx = np.where(records[i, :])[0]
            for val in idx:
                f.write('%d ' % (val,))
            f.write('%d\n' % labels[i, 1])
        f.close()
    if new_itemset_file is None:
        return
    with open(new_itemset_file, 'w') as f:
        for name in record_names:
            idx = [i for i, n in enumerate(unique_record_names) if n in name]
            for j, val in enumerate(idx):
                f.write('%d' % (val,))
                if j < len(idx) - 1:
                    f.write(' ')
            f.write('\n')
        f.close()
    with open(new_itemset_name_file, 'w') as f:
        f.write('#label_name\n')
        f.write('%s\n' % (label_names[1],))
        f.write('#itemset_name\n')
        for name in record_names:
            f.write('%s\n' % (name,))
        f.close()

def computeSTD(aList):
    variance = 0
    moyenne = average(aList)
    for aVal in aList:
        variance = variance + ((aVal - moyenne)*(aVal - moyenne))
    variance = variance / len(aList)
    return math.sqrt(variance)

class CORELS(object): # This is a simplified version of the CORELS object proposed by Ulrich
    def __init__(self):
        self.rule_ = []
        self.rule_description_ = []
        self.pred_ = []
        self.pred_description_ = []
        self.obj_ = np.inf
        self.finalAcc_ = np.inf
        self.finalFair_ = np.inf
        self.featuresNum = {}
        self.DFOk = False
        self.posPrediction = -1
    def attachFrame(self, originalDF):
        i = -1
        for col in originalDF.columns: 
            if i >= 0:
                self.featuresNum[col] = i
            i = i + 1
        self.DFOk = True
    def treatCORELSraw(self, result_file, itemset_file, itemset_name_file):       
        itemsets, data_name, label_name = load_itemsets_and_names(itemset_file, itemset_name_file)                 
        with codecs.open(result_file, 'r', 'utf-8', 'ignore') as f:
            flg = False # True when we detect the beginning of the rule list
            for line in f:
                if 'final min_objective' in line:
                    self.obj_ = float(line.split(':')[1].strip())
                if 'final accuracy' in line:
                    self.finalAcc_ = float(line.split(':')[1].strip())
                if 'final statistical parity' in line:
                    self.finalFair_ = float(line.split(':')[1].strip())
                if 'OPTIMAL RULE LIST' in line:
                    flg = True
                if flg:
                    if 'if (1) then' in line:
                        lines = line.split()
                        self.rule_description_.append(np.nan)
                        self.rule_.append(set([]))
                        self.pred_description_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_description_[-1])
                    elif 'else if' in line:
                        lines = line.split()
                        self.rule_description_.append(lines[2][1:-1])
                        idx = np.where([(self.rule_description_[-1].replace('{', '').replace('}', '') == name) for name in data_name])[0]
                        self.rule_.append(itemsets[int(idx)])
                        self.pred_description_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_description_[-1])
                    elif 'if' in line:
                        lines = line.split()
                        self.rule_description_.append(lines[1][1:-1])               
                        idx = np.where([(self.rule_description_[-1].replace('{', '').replace('}', '') == name) for name in data_name])[0]                
                        self.rule_.append(itemsets[int(idx)])
                        self.pred_description_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_description_[-1])
                    elif 'else' in line:
                        lines = line.split()
                        self.rule_description_.append(np.nan)
                        self.rule_.append(set([]))
                        self.pred_description_.append(lines[-1][1:-1])
                        self.pred_.append('True' in self.pred_description_[-1])
    def predict(self, data_file): # data_file doit en fait être le .values d'un Panda Dataframe
        resList = []
        errCnt = 0
        totPos = 0
        totNeg = 0
        totalSize = len(data_file)
        if self.DFOk:
            for aSample in range(len(data_file)): # Iterate on samples
                for i in range(len(self.rule_description_)):
                    if i == len(self.pred_description_) - 1: # if default decision
                        pred = self.pred_description_[i][1:-1]
                        if pred == self.posPrediction:
                            pred = 1
                            totPos = totPos + 1
                        else:
                            pred = 0
                            totNeg = totNeg + 1
                        resList.append(pred)
                    else:
                        ruleL = self.rule_description_[i].replace('{', '').replace('}', '').split(',')
                        match = 1
                        pred = self.pred_description_[i][1:-1]
                        for j in range(len(ruleL)):
                            index = self.featuresNum.get(ruleL[j])
                            if((data_file[aSample][index]) == 0):
                                match = 0
                            elif (data_file[aSample][index]) != 1:
                                match = 0
                                if(errCnt < 5):
                                    print("[WARNING] Found _unexpected non binary value_ : _", data_file[aSample][index], "_ for attribute: ", ruleL[j], "(num ",index,")")
                                errCnt = errCnt + 1
                        if match == 1:
                            if pred == self.posPrediction:
                                pred = 1
                                totPos = totPos + 1
                            else:
                                pred = 0
                                totNeg = totNeg + 1
                            resList.append(pred)
                            break
        else:
            print("[ERROR] Original dataframe not specified\n")
        if errCnt > 0:
            print("[...] Encountered %d unexpected values" %errCnt)
        if len(resList) != totalSize:
            print("[WARNING] preds vector not full (%d / %d)" %(len(resList), totalSize))
        else:
            print('.', end='', flush=True)
        return np.asarray(resList)


def getAcc(featuresDict, dataMatrix, model):
    pred_list = model.pred_description_
    totalInstances = len(dataMatrix[0])-1
    goodPreds = 0
    for k in range(1,totalInstances+1): # We loop on each individual in the set
        realLabel = dataMatrix[len(dataMatrix)-1][k] # Correspond à la colonne >50K pour adult
        found = 0
        error = 0
        for i in range(len(pred_list)): 
            if i == len(model.pred_description_) - 1: # if default decision
                pred = pred_list[i][1:-1]
                if pred==dataMatrix[len(dataMatrix)-1][0]:
                    pred = 1
                else:
                    pred = 0
                if int(realLabel) == int(pred):
                    goodPreds = goodPreds + 1
                    found = 1
                else:
                    error = 1
                #print(pred)           
                #print("default ->", pred)
            else:
                pred = pred_list[i][1:-1]
                if pred==dataMatrix[len(dataMatrix)-1][0]:
                    pred = 1
                else:
                    pred = 0
                #print(pred)        
                ruleL = model.rule_description_[i].replace('{', '').replace('}', '').split(',')
                correct = 1
                for i in range(len(ruleL)):
                    if int(dataMatrix[featuresDict.get(ruleL[i])][k]) == 0:
                        correct = 0
                if correct == 1: # all conditions matched
                    if int(realLabel) == int(pred):            
                        goodPreds = goodPreds + 1
                        found = 1
                    else:
                        error = 1
            if found == 1 or error == 1:
                break
    #print("Good predictions : ", goodPreds, " out of a total of ", totalInstances)
    return (float) (goodPreds/totalInstances)

def getFair(featuresDict, dataMatrix, model, protectedAttrColumn, unprotectedAttributeColumn):
    pred_list = model.pred_description_
    totalInstances = len(dataMatrix[0])-1
    totalProtected = 0
    totalUnProtected = 0
    positiveProtected = 0
    positiveUnProtected = 0
    #print("protected attribute is: ", dataMatrix[protectedAttrColumn][0])
    if unprotectedAttributeColumn == -1 :
        #print("unprotected attribute is: not(", dataMatrix[protectedAttrColumn][0], ")")
        for k in range(1,totalInstances+1): # We loop on each individual in the set
            #realLabel = dataMatrix[len(dataMatrix)-1][k] # Correspond à la colonne >50K pour adult
            found = 0
            for i in range(len(pred_list)): 
                if i == len(model.pred_description_) - 1: # if default decision
                    pred = pred_list[i][1:-1]
                    if pred==dataMatrix[len(dataMatrix)-1][0]:
                        if int(dataMatrix[protectedAttrColumn][k]) == 1:
                            positiveProtected = positiveProtected + 1
                        else:
                            positiveUnProtected = positiveUnProtected + 1
                    if int(dataMatrix[protectedAttrColumn][k]) == 1:
                        totalProtected = totalProtected + 1
                    else:
                        totalUnProtected = totalUnProtected + 1
                    found = 1
                else:
                    pred = pred_list[i][1:-1]  
                    ruleL = model.rule_description_[i].replace('{', '').replace('}', '').split(',')
                    correct = 1
                    for i in range(len(ruleL)):
                        if int(dataMatrix[featuresDict.get(ruleL[i])][k]) == 0:
                            correct = 0 
                    if correct == 1:
                        if pred==dataMatrix[len(dataMatrix)-1][0]:
                            if int(dataMatrix[protectedAttrColumn][k]) == 1:
                                positiveProtected = positiveProtected + 1
                            else:
                                positiveUnProtected = positiveUnProtected + 1
                        if int(dataMatrix[protectedAttrColumn][k]) == 1:
                            totalProtected = totalProtected + 1
                        else:
                            totalUnProtected = totalUnProtected + 1
                        found = 1
                if found == 1:
                    break
    else:
        #print("unprotected attribute is: ", dataMatrix[unprotectedAttributeColumn][0])
        for k in range(1,totalInstances+1): # We loop on each individual in the set
            #realLabel = dataMatrix[len(dataMatrix)-1][k] # Correspond à la colonne >50K pour adult
            found = 0
            for i in range(len(pred_list)):
                if i == len(model.pred_description_) - 1: # if default decision
                    pred = pred_list[i][1:-1]
                    if pred==dataMatrix[len(dataMatrix)-1][0]:
                        if int(dataMatrix[protectedAttrColumn][k]) == 1:
                            positiveProtected = positiveProtected + 1
                        elif int(dataMatrix[unprotectedAttributeColumn][k]) == 1:
                            positiveUnProtected = positiveUnProtected + 1
                    if int(dataMatrix[protectedAttrColumn][k]) == 1:
                        totalProtected = totalProtected + 1
                    elif int(dataMatrix[unprotectedAttributeColumn][k]) == 1:
                        totalUnProtected = totalUnProtected + 1
                    found = 1
                else:
                    pred = pred_list[i][1:-1]  
                    ruleL = model.rule_description_[i].replace('{', '').replace('}', '').split(',')
                    correct = 1
                    for i in range(len(ruleL)):
                        if int(dataMatrix[featuresDict.get(ruleL[i])][k]) == 0:
                            correct = 0 
                    if correct == 1:
                        if pred==dataMatrix[len(dataMatrix)-1][0]:
                            if int(dataMatrix[protectedAttrColumn][k]) == 1:
                                positiveProtected = positiveProtected + 1
                            elif int(dataMatrix[unprotectedAttributeColumn][k]) == 1:
                                positiveUnProtected = positiveUnProtected + 1
                        if int(dataMatrix[protectedAttrColumn][k]) == 1:
                            totalProtected = totalProtected + 1
                        elif int(dataMatrix[unprotectedAttributeColumn][k]) == 1:
                            totalUnProtected = totalUnProtected + 1
                        found = 1
                if found == 1:
                    break
    #print("pos prot = ", positiveProtected, " total prot = ", totalProtected, "pos unprot = ", positiveUnProtected, " total unprot = ", totalUnProtected)
    #print("Good predictions : ", goodPreds, " out of a total of ", totalInstances)
    positiveProtectedRate = (float) (positiveProtected / totalProtected)
    positiveUnProtectedRate = (float) (positiveUnProtected / totalUnProtected)
    negativeProtectedRate = (float) ((totalProtected - positiveProtected) / totalProtected)
    negativeUnProtectedRate = (float) ((totalUnProtected - positiveUnProtected) / totalUnProtected)
    return (float) (1-(abs(positiveUnProtectedRate-positiveProtectedRate)))#+abs(negativeUnProtectedRate - negativeProtectedRate)))