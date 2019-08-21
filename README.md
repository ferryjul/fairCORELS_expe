# fairCORELS
This repository contains the code we used for our experiments presented in the paper "Learning fair rule lists"

The file `genParetoCrossValMultithrV2.py` can be used to build the Pareto Front using grid search on Lambda.
The file `genParetoCrossValMultithrV2.py` can be used to build the Pareto Front using the Hyperopt library to automatically optimize the value of Lambda.
Usage :
> genParetoCrossValMultithrVx.py dataset_name -p protected_attribute_column

Where :
* `x` is either `2` or `3`
* `dataset_name` can be either `adult`, `compas`, `german_credit`, or `default_credit`
* `protected_attribute_column` is the index of the sensitive attribute column (standard values for each dataset are indicated in the scripts' comments)

NB : All other parameters (maximum number of nodes to be explored, values of Lambda for grid search, maximum number of iterations allowed for Hyperopt, unsensitive attribute column or specification, etc.) can be set easily at the beginning of each script's body.

The file `audit_mdl.py` can be used to produce a feature dependence analysis of a given model.
All parameters and file requirements can be found in the script's comments.
