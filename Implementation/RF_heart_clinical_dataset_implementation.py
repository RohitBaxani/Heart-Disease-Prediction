import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset

dataset=pd.read_csv('heart_failure_clinical_records_dataset.csv')
X=dataset.iloc[:,:2].values
Y=dataset.iloc[:,-1].values

print(dataset.head())
print(dataset.shape)
#print(dataset.info)

#SPLITTING THE DATASET
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X,y=make_classification(random_state=1)
X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test[:21])
print(y_train)
print(y_test)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
from sklearn import *
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train[0:10])
print(X_test[0:5])

#TRAINING THE RANDOM FOREST CLASSIFICATION MODEL ON THE TRAINING SET

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators= 16,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#PREDICTING A NEW RESULT
#print(classifier.predict(sc.transform([[1.14660924,0.85504438,0.02346013, -0.77698483, -0.32958246, -0.54421044,
  # 1.00358675,  1.37194498, -0.90164029, -0.09165422,  1.37456343,  1.07464806,
  #-0.62398936, -0.87378553, -1.11206697, -1.54996965, -0.07002497, -3.20705967,
  #-0.21601232, -0.08126969]])))

#PREDICTING THE TEST SET RESULTS

y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#MAKING THE CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_auc_score,roc_curve
#y_probabilities = dataset.predict(X_test)[:,1]
false_positive_rate, true_positive_rate, threshold_knn = roc_curve(y_test,y_pred)
plt.figure(figsize=(10,6))
plt.title('ROC for random forest')
plt.plot(false_positive_rate, true_positive_rate, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_pred)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
