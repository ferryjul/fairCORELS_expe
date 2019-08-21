from hyperopt import fmin, tpe, hp
from utils import *
from joblib import Parallel, delayed         
import sys

def computeObjectiveWithCV(lambdaFactor):
        l = lambdaFactor[0]
        res = computeCORELSCross(mode, hParam, kFold, l, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
        #print("1 - res[wantedIndex] = %lf\n" %(1 - res[wantedIndex]))
        return (1 - res[wantedIndex])

def runHyperOpt():
    bestLam = fmin(fn=computeObjectiveWithCV,
    space=[hp.uniform('lambdaFactor', 0.00001, 0.05)],
    algo=tpe.suggest,
    max_evals=max_evals)
    return bestLam

# Vérification des paramètres passés en ligne de commande
if len(sys.argv) < 3:
    print("usage : python save_rule.py dataset_name -p protected_attribute_column [-r lambda]")
    sys.exit()
else:
    if (sys.argv[1] != "compas") and (sys.argv[1] != "adult") and (sys.argv[1] != "german_credit") and (sys.argv[1] != "default_credit"):
        print("unknown dataset %s" %sys.argv[1])
        sys.exit()
    else:
        if sys.argv[2] != "-p":
            print("usage : python save_rule.py dataset_name -p protected_attribute_column [-r lambda]")
            sys.exit()
        if len(sys.argv) == 5:
            if ((sys.argv[4] != "-r")) and (sys.argv[4] != "-b"):
                print("usage : python save_rule.py dataset_name -p protected_attribute_column [-r lambda]")
                sys.exit()
            else:
                if sys.argv[4] == "-r":
                    lambdaFactor = sys.argv[5]
                else:
                    beta = sys.argv[5]
        if len(sys.argv) == 7:
            if sys.argv[4] != "-r":
                print("usage : python save_rule.py dataset_name -p protected_attribute_column [-r lambda]")
                sys.exit()
            else:
                lambdaFactor = sys.argv[5]
                if sys.argv[6] != "-z":
                    print("usage : python save_rule.py dataset_name -p protected_attribute_column [-r lambda]")
                    sys.exit()
                else:
                    beta = sys.argv[7]

'''
For adult, discrimination Female/Male :
fairnessProtectedFeature => 2 //19
fairnessUnprotectedFeature => 1 //20

For compas, discrimination race:african-american/race:caucasian :
fairnessProtectedFeature => 8
fairnessUnprotectedFeature => 10

For German Credit Dataset, discrimination age__<27/others :
fairnessProtectedFeature => 48
fairnessUnprotectedFeature => -1

For Default Credit Dataset, discrimination male/female :
fairnessProtectedFeature => 5
fairnessUnprotectedFeature => 6
'''

fairnessProtectedFeature = int(sys.argv[3])
maxNBnodes = 100000
fairnessUnprotectedFeature = 20
protAttrLabel = ""
unprotAttrLabel = ""
beta = 0.0
kFold = 10
NBPoints = 4
max_evals = 2
# --- Input files : -----------------------
# Default dataset is Adult
dataset_name = "adult"
itemSetFile = "./data/adult/adult_itemset.txt"
itemSetNamesFile =  "./data/adult/adult_itemset_name.txt"
fullFeatures = "./data/adult/adult_full.feature"
fullLabels = "./data/adult/adult_full.label"
trainMinor = ""
if sys.argv[1] == "compas":
    dataset_name = "compas"
    itemSetFile = "./data/compas/compas_itemset.txt"
    itemSetNamesFile =  "./data/compas/compas_itemset_name.txt"
    fullFeatures = "./data/compas/compas_full.feature"
    fullLabels = "./data/compas/compas_full.label"
    trainMinor = ""
    fairnessUnprotectedFeature = 10
elif sys.argv[1] == "german_credit":
    dataset_name = "german_credit"
    itemSetFile = "./data/german_credit/german_credit_itemset.txt"
    itemSetNamesFile =  "./data/german_credit/german_credit_itemset_name.txt"
    fullFeatures = "./data/german_credit/german_credit_full.feature"
    fullLabels = "./data/german_credit/german_credit_full.label"
    trainMinor = ""
    fairnessUnprotectedFeature = -1
elif sys.argv[1] == "default_credit":
    dataset_name = "default_credit"
    itemSetFile = "./data/default_credit/default_credit_itemset.txt"
    itemSetNamesFile =  "./data/default_credit/default_credit_itemset_name.txt"
    fullFeatures = "./data/%s/default_credit_full.feature" %dataset_name
    fullLabels = "./data/%s/default_credit_full.label" %dataset_name
    trainMinor = ""
    fairnessUnprotectedFeature = 6
# --- Output files : -----------------------
CORELS_raw_file = "./gen_files/%s/corels_raw.txt" %dataset_name
modelFile = "./gen_files/%s/model_dumped.mdl" %dataset_name
output_file = "./gen_files/%s/model_analysis.txt" %dataset_name
# -----------------------------------------------------------------------------------
if fairnessUnprotectedFeature == -1 :
    unprotArg = ""
else:
    unprotArg = "-g %d" %fairnessUnprotectedFeature


# Préparation de la matrice et du dictionnaire
print("Préparation des matrices de données et du dictionnaire")
fullMatrix = build_matrix(fullFeatures, fullLabels)
featuresDict = build_dictionnary(fullFeatures) # Train ou test, peu importe ici vu qu'on ne regarde que les "headers"
print("..............................")

protAttrLabel = fullMatrix[fairnessProtectedFeature-1][0]
if fairnessUnprotectedFeature == -1:
    unprotAttrLabel = "not(%s)" %protAttrLabel
else:
    unprotAttrLabel = fullMatrix[fairnessUnprotectedFeature-1][0]
# Calculs des indices de découpage
print("Nombre d'attr + labels : %d, Nombre d'instances : %d" % (len(fullMatrix),len(fullMatrix[0])-1))
setSize = (len(fullMatrix[0])-1) / kFold
print("set size = %d" %setSize)

# Génération des différents fichiers train/tests (sets)
print("Génération des différents ensembles")
for i in range(kFold):
    trainFeatFileName = "_tmp/temp_train_features_%d.feature" % (i)
    testFeatFileName = "_tmp/temp_test_features_%d.feature" % (i)
    trainLabelsFileName = "_tmp/temp_train_labels_%d.label" % (i)
    testLabelsFileName = "_tmp/temp_test_labels_%d.label" % (i)
    with open(trainFeatFileName,'w') as trainSetFeat:
        with open(testFeatFileName,'w') as testSetFeat:
            with open(trainLabelsFileName,'w') as trainSetLabs:
                with open(testLabelsFileName,'w') as testSetLabs:
                    for f in range(len(fullMatrix)): #loop on features
                        for j in range(len(fullMatrix[0])): #loop on instances
                            if f >= len(fullMatrix) - 2: #Labels
                                if j == 0:
                                    trainSetLabs.write("{%s}" %(fullMatrix[f][j]))
                                    testSetLabs.write("{%s}" %(fullMatrix[f][j]))
                                else:
                                    if j <= (i*setSize) or j >= ((i+1)*setSize): # training set
                                        trainSetLabs.write(" %d" %(int(fullMatrix[f][j])))
                                    else: # test set
                                        testSetLabs.write(" %d" %(int(fullMatrix[f][j])))
                            else: #Features
                                if j == 0:
                                    trainSetFeat.write("{%s}" %(fullMatrix[f][j]))
                                    testSetFeat.write("{%s}" %(fullMatrix[f][j]))
                                else:
                                    if j <= (i*setSize) or j >= ((i+1)*setSize): # training set
                                        trainSetFeat.write(" %d" %(int(fullMatrix[f][j])))
                                    else: # test set
                                        testSetFeat.write(" %d" %(int(fullMatrix[f][j])))
                        if f >= len(fullMatrix) - 2: #Labels
                            trainSetLabs.write("\n")
                            testSetLabs.write("\n")
                        else:
                            trainSetFeat.write("\n")
                            testSetFeat.write("\n")
                testSetLabs.close()
            trainSetLabs.close()
        testSetFeat.close()
    trainSetFeat.close()

print("Calcul des n plus proches voisins pour les k sets")
nbNeigh = 10
kClosestDict = Parallel(n_jobs=-1)(delayed(computekNN)(kF = kNb, k = nbNeigh) for kNb in range(kFold))

print("..............................")
trainingF= []
trainingA = []
testingF = []
testingA = []
accSTD = []
fairSTD = []
models = []
lambdaList = []
incons = []

# Initial solution, for maximum fairness :
print("Initial max fairness solution generation...")
mode = 2
hParam = ""
wantedIndex = 1
bestLambda = runHyperOpt().get('lambdaFactor')
results = computeCORELSCross(2, "", kFold, bestLambda, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile, kClosestDict)
lambdaList.append(bestLambda)
incons.append(results[7])
models.append(results[6])
fairSTD.append(results[5])
accSTD.append(results[4])
trainingF.append(results[3])
trainingA.append(results[2])
testingF.append(results[1])
testingA.append(results[0])
print("lambda = %lf, train fair = %lf, train acc = %lf, test fair = %lf, test acc = %lf" %(bestLambda, trainingF[0], trainingA[0], testingF[0], testingA[0]))

# Initial solution, for maximum accuracy :
print("Initial max accuracy solution generation...")
mode = 4
hParam = ""
wantedIndex = 0
bestLambda = runHyperOpt().get('lambdaFactor')
results = computeCORELSCross(4, "", kFold, bestLambda, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
lambdaList.append(bestLambda)
incons.append(results[7])
models.append(results[6])
fairSTD.append(results[5])
accSTD.append(results[4])
trainingF.append(results[3])
trainingA.append(results[2])
testingF.append(results[1])
testingA.append(results[0])
print("lambda = %lf, train fair = %lf, train acc = %lf, test fair = %lf, test acc = %lf" %(bestLambda, trainingF[1], trainingA[1], testingF[1], testingA[1]))
print("..............................")
diff = abs(trainingF[0] - trainingF[1])
fairn = trainingF[0]
foldI = 0
while fairn > trainingF[1]:
    print("epsilon-constraint = %lf\n" %fairn)
    hParam = "-h %lf" %fairn
    # use hyperopt with tree of Parzen estimators to find the best Lambda value
    mode = 3
    wantedIndex = 0
    bestLambda = runHyperOpt().get('lambdaFactor')
    results = computeCORELSCross(3, hParam, kFold, bestLambda, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
    lambdaList.append(bestLambda)
    incons.append(results[7])
    models.append(results[6])
    fairSTD.append(results[5])
    accSTD.append(results[4])
    trainingF.append(results[3])
    trainingA.append(results[2])
    testingF.append(results[1])
    testingA.append(results[0])
    print("lambda = %lf, train fair = %lf, train acc = %lf, test fair = %lf, test acc = %lf" %(bestLambda, trainingF[foldI+2], trainingA[foldI+2], testingF[foldI+2], testingA[foldI+2]))
    print("=== %lf/100 done === \n" %(100*(foldI+1)/NBPoints))
    fairn = fairn - (diff/NBPoints)
    foldI = foldI + 1
# Elimination des solutions dominées
print("Cleaning dominated solutions and duplicates...")
testingA, testingF, accSTD, fairSTD, models, lambdaList, incons = clean_lists_with_lambda(testingA, testingF, accSTD, fairSTD, models, lambdaList, incons)
plt.scatter(testingA, testingF, label = "Testing, lambda auto optimized")

# Add title and axis names
plt.title("Pareto Front approximation (dataset = %s,\nProtected attribute = %s Unprotected attribute = %s)" %(dataset_name, protAttrLabel, unprotAttrLabel))
plt.xlabel('Accuracy')
plt.ylabel('Fairness')

plt.legend(loc='lower left')

plt.autoscale(tight=True)
#plt.show()
plt.savefig("./plots/result_plot.png", aspect='auto')
with open('./plots/results.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Lambda', 'Fairness', 'Accuracy', 'Unfairness', 'Delta', 'FairnessSTD', 'AccuracySTD', 'Inconsistency', 'Best model'])
    for i in range(len(testingA)):
        csv_writer.writerow([lambdaList[i], testingF[i], testingA[i], 1-testingF[i], testingA[i] - (1-testingF[i]) ,fairSTD[i], accSTD[i], incons[i], models[i]])