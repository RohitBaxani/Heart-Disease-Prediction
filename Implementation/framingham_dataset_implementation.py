import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

#DATA PRE-PROCESSING
#Collecting the data
#1a
heart_data=pd.read_csv("framingham.csv")
heart_data.head(10)
#print(heart_data.head(10))

heart_data.TenYearCHD.value_counts() # df.target.unique()
disease = len(heart_data[heart_data['TenYearCHD'] == 1])
no_disease = len(heart_data[heart_data['TenYearCHD']== 0])

#1b
#print("# of patients in original data:" +str(len(heart_data.index)))

#Analyzing Data
value=(heart_data['TenYearCHD']=='1')&(heart_data['sex']=='1') #0 means female and 1 means male
#heart_data['color']= np.where( value==True, "#0000FF", "#FFC0CB")

sns.countplot(x='TenYearCHD',data=heart_data)
plt.show()
sns.countplot(x="TenYearCHD",hue="sex",data=heart_data)
plt.show()
sns.countplot(x="TenYearCHD",hue="currentSmoker",data=heart_data)
plt.show()

heart_data["age"].plot.hist(bins=10,figsize=(10,5))
plt.show()

heart_data["totChol"].plot.hist(bins=20,figsize=(10,5))
plt.show()

print(heart_data.info())

#DATA WRANGLING
print(heart_data.isnull())
print(heart_data.isnull().sum())

sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

sns.boxplot(x="heartRate",y="BMI",data=heart_data)
plt.show()

print(heart_data.head(5))
heart_data.drop("glucose",axis=1,inplace=True)
print(heart_data.head(5))

heart_data.dropna(inplace=True)
sns.heatmap(heart_data.isnull(),yticklabels=False,cbar=False)
plt.show()

print(heart_data.isnull().sum())
#pd.get_dummies(heart_data[''],drop_first=True)

#TRAIN DATA
X=heart_data.drop("TenYearCHD",axis=1)
y=heart_data["TenYearCHD"]

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X,y=make_classification(random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pipe=make_pipeline(StandardScaler(),LogisticRegression())
print(pipe.fit(X_train,y_train))
#ACCURACY CHECK 
predictions=pipe.predict(X_test)
R=classification_report(y_test,predictions)
print(R)
print(pipe.score(X_test,y_test))
print(confusion_matrix(y_test,predictions))

#PIE CHART
y = ('Heart Disease', 'No Disease')
y_pos = np.arange(len(y))
x = (disease, no_disease)
labels = 'Heart Disease', 'No Disease'
sizes = [disease, no_disease]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%', startangle=90) 
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of target', size=16)
plt.show() # Pie chart, where the slices will be ordered and plotted counter-clockwise:


#Correlated features of dataset

top = 15
corr = heart_data.corr()
top15 = corr.nlargest(top, 'TenYearCHD')['TenYearCHD'].index
corr_top15 = heart_data[top15].corr()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_top15, square=True, ax=ax, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size':12})
plt.title('Top correlated features of dataset', size=16)
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,predictions)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for logistic regression')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
