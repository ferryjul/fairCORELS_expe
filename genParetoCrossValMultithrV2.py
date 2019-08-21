from utils import *
from joblib import Parallel, delayed         


# Vérification des paramètres passés en ligne de commande (pour l'instant manuelle mais à changer)
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
maxNBnodes = 100000
fairnessProtectedFeature = int(sys.argv[3])
fairnessUnprotectedFeature = 20
protAttrLabel = ""
unprotAttrLabel = ""
beta = 0.0
kFold = 10
NBPoints = 3
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

trainingF= []
trainingA = []
testingF = []
testingA = []
accSTD = []
fairSTD = []
models = []
incons = []
lambdaFactors = [0.001, 0.005] # Put here the different lambdas that you want grid search to be performed with
index = 0
for lambdaFactor in lambdaFactors:
    # Initial solution, for maximum fairness :
    results = computeCORELSCross(2, "", kFold, lambdaFactor, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
    incons.append([results[7]])
    models.append([results[6]])
    fairSTD.append([results[5]])
    accSTD.append([results[4]])
    trainingF.append([results[3]])
    trainingA.append([results[2]])
    testingF.append([results[1]])
    testingA.append([results[0]])
    print("train fair = %lf, train acc = %lf, test fair = %lf, test acc = %lf" %(trainingF[index][0], trainingA[index][0], testingF[index][0], testingA[index][0]))

    # Initial solution, for maximum accuracy :
    mode = 1
    hParam = ""
    results = computeCORELSCross(4, "", kFold, lambdaFactor,  dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
    incons[index].append(results[7])
    models[index].append(results[6])
    fairSTD[index].append(results[5])
    accSTD[index].append(results[4])
    trainingF[index].append(results[3])
    trainingA[index].append(results[2])
    testingF[index].append(results[1])
    testingA[index].append(results[0])
    print("train fair = %lf, train acc = %lf, test fair = %lf, test acc = %lf" %(trainingF[index][1], trainingA[index][1], testingF[index][1], testingA[index][1]))

    diff = abs(trainingF[index][0] - trainingF[index][1])
    fairn = trainingF[index][0]
    foldI = 0
    while fairn > trainingF[index][1]:
        hParam = "-h %lf" %fairn
        results = computeCORELSCross(3, hParam, kFold, lambdaFactor, dataset_name, maxNBnodes, fairnessProtectedFeature, fairnessUnprotectedFeature, beta, unprotArg, trainMinor, itemSetFile, itemSetNamesFile,kClosestDict)
        incons[index].append(results[7])
        models[index].append(results[6])
        fairSTD[index].append(results[5])
        accSTD[index].append(results[4])
        trainingF[index].append(results[3])
        trainingA[index].append(results[2])
        testingF[index].append(results[1])
        testingA[index].append(results[0])
        print("=== %lf/100 done for iteration %d/%d === \n" %(100*(foldI+1)/NBPoints, index+1, len(lambdaFactors)))
        fairn = fairn - (diff/NBPoints)
        foldI = foldI + 1
    # Elimination des solutions dominées
    print("Cleaning dominated solutions and duplicates...")
    testingA[index], testingF[index], accSTD[index], fairSTD[index], models[index], incons[index] = clean_lists_no_lambda(testingA[index], testingF[index], accSTD[index], fairSTD[index], models[index], incons[index])
    plt.scatter(testingA[index], testingF[index], label = "Testing, lambda = %lf" %lambdaFactor)
    index = index + 1

# Add title and axis names
plt.title("Pareto Front approximation (dataset = %s,\nProtected attribute = %s Unprotected attribute = %s)" %(dataset_name, protAttrLabel, unprotAttrLabel))
plt.xlabel('Accuracy')
plt.ylabel('Fairness')


plt.legend(loc='lower left')

plt.axis([0,1,0,1])
#plt.show()
plt.autoscale(tight=True)
plt.savefig("./plots/result_plot.png")

with open('./plots/results.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Lambda', 'Fairness', 'Accuracy', 'Unfairness', 'Delta', 'FairnessSTD', 'AccuracySTD', 'Inconsistency', 'Best model'])
    index = 0
    for l in lambdaFactors:
        for i in range(len(testingA[index])):
            csv_writer.writerow([l, testingF[index][i], testingA[index][i], 1-testingF[index][i], testingA[index][i] - (1-testingF[index][i]) ,fairSTD[index][i], accSTD[index][i], incons[index][i], models[index][i]])
        index = index + 1
        csv_writer.writerow(["", "", "", "", "", ""])