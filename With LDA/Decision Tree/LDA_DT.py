from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import metrics


digit = datasets.load_iris()
x_train = digit.data
y_train = digit.target

x_test = digit.data
y_test = digit.target

print (len(x_train[0]))
lda = LinearDiscriminantAnalysis(n_components=8)
model = lda.fit(x_train, y_train).transform(x_train)
print(len(model[0]))

DT=DecisionTreeClassifier()
NEW_model=DT.fit(model,y_train)

accuracy = NEW_model.score(model, y_test)
print(accuracy)
