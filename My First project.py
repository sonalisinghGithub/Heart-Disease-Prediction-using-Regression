#!/usr/bin/env python
# coding: utf-8

# **Problem:-**
#     The "goal" field refers to the presence of heart disease in the patient.
# 
# 

# In[3]:


# 

##Loading appropriate libraries##
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.svm import SVC
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#returns the path from current working diectory and importing the dataset
print(os.getcwd())
os.chdir("J:\\My projects\\Heart disease using python")


# In[5]:


#Reading csv(comma separated values) file.......
df=pd.read_csv("heart.csv")
df.head(5)


# **Short description about each variable:-**
# 
# It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
# 
# 1.age: The person's age in years
# 
# 2.sex: The person's sex (1 = male, 0 = female)
# 
# 3.cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# 4.trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# 5.chol: The person's cholesterol measurement in mg/dl
# 
# 6.fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 7.restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# 8.thalach: The person's maximum heart rate achieved
# 
# 9.exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 10.oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# 11.slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# 12.ca: The number of major vessels (0-3)
# 
# 13.thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 14.target: Heart disease (0 = no, 1 = yes)

# In[7]:



#Here all are categorical data......
df.describe()


# In[8]:


#checking the dimensions:-----
df.shape


# In[9]:



#checking whether a dataset have a null value or not...
df.isnull().sum()


# In[11]:


#check the correlation between the classes or features....
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()


# from the above corelation plot we see that cp(chest pain),thalch and slope are highly corelated with the target.

# In[14]:


#========1.Target=========#
sns.distplot(df['target'],rug=True)
plt.show()


# In[15]:



#checking the categories present in Target variables....
df.target.value_counts()


# In[16]:


#countplot based on categories of Target variable...
sns.countplot(x="target", data=df,palette="bwr")
plt.show()


# In[18]:


#ploting a bar graph between X-Target vs Y-fbs(fasting blood sugar) and hue=thal
plt.figure(num=None, figsize=(12, 6))
# specify hue="categorical_variable"
sns.barplot(x='target', y='fbs', hue="thal", data=df)
plt.show()


# In[19]:


#ploting a bar graph between X-Target vs Y-thalach and hue=slope
plt.figure(num=None, figsize=(8, 6))
# specify hue="categorical_variable"
sns.boxplot(x='target', y='thalach', hue="slope", data=df)
plt.show()


# In[20]:


##=======Age========##
sns.distplot(df['age'],rug=True)
plt.show()


# In[24]:


min_age=min(df.age)
max_age=max(df.age)
mean_age=df.age.mean()
print(min_age,max_age,mean_age)


# In[25]:


#barplot on age between 44 to 62 age
sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.show()


# In[26]:


#=======sex=======#
df.sex.value_counts()


# In[27]:


#Distribution graph for sex features...
sns.distplot(df['sex'],rug=True)
plt.show()


# In[28]:


#countplot for sex variables as it is having two category....
sns.countplot(x='sex', data=df)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[29]:


plt.figure(num=None, figsize=(7, 4))


# In[30]:


# specify hue="sex"
sns.barplot( x='target',y='thalach',hue='sex', data=df)
plt.show()


# In[31]:


#conclusion is Women are 4 times more likely to die from heart disease than breast cancer
#we see that the rate of heart disease in females have more in comprission of male.
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5),color=['#1CA53B','#AA1111' ])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[32]:



#==========fbs: fasting blood sugar==============#

sns.countplot(x='fbs', data=df)
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl)')
plt.show()


# In[33]:


pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(10,5),color=['#FFC300','#581845' ])
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[34]:


#=========Cp: chest pain===========#
sns.distplot(df['cp'],rug=True)
plt.show()


# In[35]:


sns.countplot(x='cp', data=df)
plt.xlabel('Chest Pain Type')
plt.show()


# In[36]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[37]:


#lmplot== lmplot() combines regplot() and FacetGrid
sns.lmplot(x="trestbps", y="chol",data=df,hue="cp")
plt.show()


# In[38]:


#=====thalach: maximum heart rate achieved======#
sns.barplot(x=df.thalach.value_counts()[:10].index,y=df.thalach.value_counts()[:10].values)
plt.xlabel('max heart rate')
plt.ylabel('Counter')
plt.show()



# In[39]:



###from the dataset 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables.#

chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)
df=pd.concat([df,chest_pain],axis=1)
df.drop(['cp'],axis=1,inplace=True)
sp=pd.get_dummies(df['slope'],prefix='slope')
th=pd.get_dummies(df['thal'],prefix='thal')
frames=[df,sp,th]
df=pd.concat(frames,axis=1)
df.drop(['slope','thal'],axis=1,inplace=True)


# In[40]:



#checking the data set--here my dataset is converted into dummies...
df.head(5)


# In[42]:


#Feature selection
X = df.drop(['target'], axis = 1)
y = df.target.values
X


# In[43]:


y


# In[45]:



#Spliting the 80% of the dataset into train_data and 20% of the dataset into test_data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)



#scale your data as it is not in proper sacle
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[46]:



#Applying algorithm to the training and test data set...
#LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
print("The accuracy of  LogisticRegression is:",accuracy_score(y_test, lr_pred))


# In[47]:



#SVM classifier
svc_c=SVC(kernel='linear',random_state=0)
svc_c.fit(X_train,y_train)
svc_pred=svc_c.predict(X_test)
sv_cm=confusion_matrix(y_test,svc_pred)
print("The accuracy of  SVC is:",accuracy_score(y_test, svc_pred))


# In[48]:


#Bayes
gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
bayes_pred=gaussian.predict(X_test)
bayes_cm=confusion_matrix(y_test,bayes_pred)
print("The accuracy of naives bayes is:",accuracy_score(bayes_pred,y_test))


# In[49]:


#SVM regressor
svc_r=SVC(kernel='rbf')
svc_r.fit(X_train,y_train)
svr_pred=svc_r.predict(X_test)
svr_cm=confusion_matrix(y_test,svr_pred)
print("The accuracy of SCR is:",accuracy_score(y_test, svr_pred))


# In[50]:


#RandomForest
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
print("The accuracy of RandomForestClassifier is:",accuracy_score(rdf_pred,y_test))


# In[51]:


# DecisionTree Classifier
dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree_c.fit(X_train,y_train)
dtree_pred=dtree_c.predict(X_test)
dtree_cm=confusion_matrix(y_test,dtree_pred)
print("The accuracy of DecisionTreeClassifier is:",accuracy_score(dtree_pred,y_test))


# In[52]:



#KNN
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
knn_cm=confusion_matrix(y_test,knn_pred)
print("The accuracy of KNeighborsClassifier is:",accuracy_score(knn_pred,y_test))



# In[53]:


#confusion matrix.....
plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.title("LogisticRegression_cm")
sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,2)
plt.title("SVM_regressor_cm")
sns.heatmap(sv_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,3)
plt.title("bayes_cm")
sns.heatmap(bayes_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)
plt.subplot(2,4,4)
plt.title("RandomForest")
sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,5)
plt.title("SVM_classifier_cm")
sns.heatmap(svr_cm,annot=True,cmap="Reds",fmt="d",cbar=False)
plt.subplot(2,4,6)
plt.title("DecisionTree_cm")
sns.heatmap(dtree_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,7)
plt.title("kNN_cm")
sns.heatmap(knn_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()


# In[55]:




#ROC and Precision Recall Curve for the model which gives the heighest accuracy
def plotting(true,pred):
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    precision,recall,threshold = precision_recall_curve(true,pred[:,1])
    ax[0].plot(recall,precision,'g--')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))
    fpr,tpr,threshold = roc_curve(true,pred[:,1])
    ax[1].plot(fpr,tpr)
    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    
    

    
plt.figure()
plotting(y_test,gaussian.predict_proba(X_test))


# **Short summary:-**
# 
# We started with the data exploration where we got a feeling for the dataset, checked about missing data and learned which features are important. During this process we used seaborn and matplotlib to do the visualizations. During the data preprocessing part, we converted features into numeric ones, grouped values into categories and created a few new features. Afterwards we started training machine learning models, and applied cross validation on it. Of course there is still room for improvement, like doing a more extensive feature engineering, by comparing and plotting the features against each other and identifying and removing the noisy features. You could also do some ensemble learning.Lastly, we looked at it’s confusion matrix and computed the models precision.
# 
