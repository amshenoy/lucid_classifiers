# LUCID Classifiers Analysis
This analysis program was made so that various institutes such as CERN@School and the Institute for Research in Schools can use this program for analysing their data from Timepix or Medipix particle detectors.

![CERN@School](http://cernatschool.web.cern.ch/sites/cernatschool.web.cern.ch/files/images/logos/IRIS_logo_white-backing.JPG)
![IRIS](https://cernatschool.web.cern.ch/sites/all/themes/cern/img/cern-logo-large.png)

Models
-------------
The neural_model folder contains the neural model used for classification. The neural model can be viewed with Tensorboard with its respective accuracy and loss graphs.
![Tensorflow](https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688)

The benchmark_classifiers folder contains all the classifiers used for testing against the neural network.

LCA API
--------
```
from lucid_classifiers.analysis import classify

blob = [[0,0],[0,1],[1,0],[1,1],[0,2],[1,2],[0,3],[1,3]]

## Composite Classifier (No parameter) <- Picks the most popular prediction from all the analysis methods
print(classify(blob))

## SVM Classifier
print(classify(blob,"svm"))
## KNN Classifier
print(classify(blob,"knn"))
## Decision Tree Classifier
print(classify(blob, "dt"))
## Random Forest Classifier
print(classify(blob, "rf"))

## Neural Classifier
print(classify(blob, "neural"))

## LUCID Algorithm
print(classify(blob, "lucid"))

```

Dependencies
------------
	- Tensorflow
	- LUCID Utils
	- Numpy