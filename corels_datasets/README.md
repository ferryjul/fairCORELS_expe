 # About this repository
This repository contains input datasets for CORELS. It also include preprocesing algorithms that we used to binarize numerical attributes for **Adult Income** dataset. Datasets are in the folder named **data**. For a dataset **x**, we use **x.py** to preprocess (if required), and **x_train_binary.csv** (resp. **x_test_binary.csv**) as training (resp. test set). 

**Datasets:Folder name**
1. Adult Income : adult -> Binarized using [1]
2. Compas: compas -> Dataset provided with the original version of CORELS, originally binarized
3. German Credit : german_credit ->  Binarized using the quartiles for continuous features, and one hot encoding for categorical data
4. Default of Credit Card Clients : default_credit -> Binarized using [1] ; when [1] was not able to find a good cut, used the quartiles for continuous features, and one hot encoding for categorical data

 # Discretization MDLP
[1] Python implementation of Fayyad and Irani's MDLP criterion discretiation algorithm (Irani, Keki B. "Multi-interval discretization of continuous-valued attributes for classiﬁcation learning." (1993)). We use the code proposed in [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys/tree/master/Discretization)

# Dependencies
1. Pandas
2. Numpy
3. Sklearn
