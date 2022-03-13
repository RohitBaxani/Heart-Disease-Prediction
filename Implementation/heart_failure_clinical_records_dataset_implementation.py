import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#DATA PRE-PROCESSING
#Collecting the data
#1a
heart_data=pd.read_csv("heart_failure_clinical_records_dataset.csv")
heart_data.head(10)
print(heart_data.head(10))

heart_data.DEATH_EVENT.value_counts() # df.target.unique()
disease = len(heart_data[heart_data['DEATH_EVENT'] == 1])
no_disease = len(heart_data[heart_data['DEATH_EVENT']== 0])

#1b
print("# of patients in original data:" +str(len(heart_data.index)))

#Analyzing Data
value=(heart_data['DEATH_EVENT']=='1')&(heart_data['sex']=='1') #0 means no death and 1 means patient deceased


sns.countplot(x='DEATH_EVENT',data=heart_data)
plt.show()
sns.countplot(x="DEATH_EVENT",hue="sex",data=heart_data)
plt.show()
sns.countplot(x="DEATH_EVENT",hue="high_blood_pressure",data=heart_data)
plt.show()
heart_data["age"].plot.hist(bins=10,figsize=(10,5))
plt.show()
heart_data["creatinine_phosphokinase"].plot.hist(bins=20,figsize=(10,5))
plt.show()
print(heart_data.info())

#DATA WRANGLING
print(heart_data.isnull())
print(heart_data.isnull().sum())

sns.heatmap(heart_data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

#sns.boxplot(x="age",y="ejection_fraction",linewidth=2,data=heart_data)
heart_data["ejection_fraction"].plot.hist(bins=20,figsize=(10,5))
plt.show()
print(heart_data.head(5))

#DATA CLEANING
heart_data.dropna(inplace=True)
sns.heatmap(heart_data.isnull(),yticklabels=False,cbar=False)
plt.show()

#TRAIN DATA
X=heart_data.drop("DEATH_EVENT",axis=1)
y=heart_data["DEATH_EVENT"]

X,y=make_classification(random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

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
top15 = corr.nlargest(top, 'DEATH_EVENT')['DEATH_EVENT'].index
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
