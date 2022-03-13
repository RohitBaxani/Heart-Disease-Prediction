#Dataset for prediction of heart attack
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import metrics

#DATA PRE PROCESSING
#Collecting the data
heart_data=pd.read_csv("heart.csv")
heart_data.head(10)
print(heart_data.head(10))

#1b
print("# of patients in original data:"+str(len(heart_data.index)))

#Analyzing Data
value=(heart_data['output']=='1')&(heart_data['sex']=='1')
#heart_data['color']=np.where(value==True,"#0000FF","#FFC0CB")

sns.countplot(x='sex',data=heart_data)
plt.show()
sns.countplot(x='output',hue='sex',data=heart_data)
plt.show()
sns.countplot(x='output',hue='cp',data=heart_data)
plt.show()

heart_data["age"].plot.hist(bins=10,figsize=(10,5))
plt.show()

heart_data["trtbps"].plot.hist(bins=20,figsize=(10,5))
plt.show()

print(heart_data.info())

#DATA WRANGLING
print(heart_data.isnull())
print(heart_data.isnull().sum())

sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

sns.boxplot(x="chol",y="thalachh",linewidth=2,data=heart_data)
plt.show()

print(heart_data.head(5))

#DATA CLEANING
heart_data.dropna(inplace=True)
sns.heatmap(heart_data.isnull(),yticklabels=False,cbar=False)
plt.show()

#TRAIN AND TEST DATA
X=heart_data.drop("output",axis=1)
y=heart_data["output"]


X,y=make_classification(random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#EXECUTION

cls= svm.SVC(kernel="linear")
#train the model
cls.fit(X_train,y_train)
#predict the response
pred= cls.predict(X_test)
#Creating the Confusion matrix
cm=confusion_matrix(y_test,y_pred=pred)
print(cm)
print("accuracy:",metrics.accuracy_score(y_test,y_pred=pred))
#precision score
print("precision:",metrics.precision_score(y_test,y_pred=pred,average=None))
#recall score
print("recall",metrics.recall_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test,y_pred=pred))

#VISULAIZING THE TRAINING SET RESULT
support_vectors_per_class = cls.n_support_
print(support_vectors_per_class)
support_vectors = cls.support_vectors_
plt.scatter(X_train[:,0], X_train[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#VISUALIZING THE TEST SET RESULT
plt.scatter(X_test[:,0], X_test[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


