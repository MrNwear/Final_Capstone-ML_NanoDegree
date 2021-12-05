#!/usr/bin/env python
# coding: utf-8

# # Spam Detection Project
# ## Machine learning nano-degree Capstone project
# ### Yosef Hesham Nwear

# ## 1- Import libraries load the data

# In[5]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')

# load data set for csv file 
data=pd.read_csv('SMSSpamCollection.csv',encoding='latin-1',usecols=[0,1])


# In[6]:


print(data.head(),'\n')
print(data.info(),'\n')
print(data.describe())


# In[7]:


data.groupby('v1').describe()


# In[8]:


# Visualization of the most freq words in the dataset 
cunt1=Counter(" ".join(data['v2']).split()).most_common(30)
df1 =pd.DataFrame.from_dict(cunt1)
df1=df1.rename(columns={0:'word',1:'count'})
fig=plt.figure()
ax=fig.add_subplot()
df1.plot.bar(ax=ax,legend=False)
xticks = np.arange(len(df1['word']))
ax.set_xticks(xticks)
ax.set_xticklabels(df1['word'])
ax.set_ylabel('Frequency : Number of occurence')
plt.show()


# In[9]:


# Count the number of words in each Text
data['Count']=0
for i in np.arange(0,len(data.v2)):
    data.loc[i,'Count'] = len(data.loc[i,'v2'])

print(data.head(),'\n')

# Unique values in target set
print ("Unique values in the Class set: ", data['v1'].unique())


# In[10]:


data=data.replace(['ham','spam'],[0,1])


# In[11]:


# Collecting ham messages 
ham=data[data.v1==0]
ham_count  = pd.DataFrame(pd.value_counts(ham['Count'],sort=True).sort_index())
print ("Number of ham messages in data set :", ham['v1'].count(),'\n')
print ("Ham Count value :", ham_count['Count'].count(),'\n')
print(ham.head())


# In[12]:


# Collecting spam messages
spam=data[data.v1 == 1]
spam_count=pd.DataFrame(pd.value_counts(spam['Count'],sort=True).sort_index())
print("Number of spam messages in data set :" ,spam['v1'].count(),'\n')
print("Spam count value :",spam_count['Count'].count(),'\n')
print(spam.head())


# In[ ]:





# ## 2- Preprocess the data

# In[13]:


from sklearn.preprocessing import LabelEncoder

labels=data['v1']

#convert class labels to binary values spam : 1 , ham : 0

encoder=LabelEncoder()
Y=encoder.fit_transform(labels)

print(Y[:10])


# In[14]:


# collecting text messages to clear and deal with it 

text_messages=data['v2']


# In[15]:


# clear text data using reguler expressions

# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')


# In[16]:


# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


# In[17]:


# change capital words  to lower case 

processed=processed.str.lower()
processed.head()


# In[ ]:


# use NLP toolkit "NLTK"

import nltk
nltk.download('stopwords')


# In[19]:


# Cleaning data from stop words

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#if true it will download all the stopwords
if False:
    nltk.download('stopwords')

#if true will create vectorizer without any stopwords
if False:
    vectorizer = TfidfVectorizer()

#if true will create vectorizer with stopwords
if True:
    stopset = set(stopwords.words("english"))
    vectorizer = TfidfVectorizer(stop_words=stopset,binary=True)


# In[20]:


# Extract feature column 'Text'
X=vectorizer.fit_transform(processed)
# Extract target column 'label'
y=data['v1']


# In[ ]:





# ## 3- Prediction

# In[21]:


# split data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[22]:


print(X_train.shape[0])
print(y_test.shape[0])


# In[33]:


# use models  form sklearn to predict

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

objects=( 'DTs', 'LOgisticRegression','Multi-NB', 'SVM','KNN')


# In[34]:


# Training and predicting 

# function to train model
def train_classifier(clf, X_train, y_train):    
    clf.fit(X_train, y_train)
    
# function to predict features 
def predict_labels(clf, features):
    return (clf.predict(features))


# In[35]:


# initialize Classifiers
DecisionTree = DecisionTreeClassifier(random_state=42)
LogisticRegression= LogisticRegression()
NaiveBayse = MultinomialNB(alpha=1.0,fit_prior=True)
SVC = SVC(kernel = 'linear')
KNeighborsClassifier=KNeighborsClassifier(n_neighbors=100)


clf = [DecisionTree,LogisticRegression,NaiveBayse,SVC,KNeighborsClassifier]
names =['DecisionTree','LogisticRegression','NaiveBayse','SVC','KNN']
scores=[]
for j in range(0,5):
    scores.append(cross_val_score(clf[j], X_train, y_train, cv=5))
    print ("{} : {}".format(names[j],scores[j]))


# In[36]:


# train & predict models and calculate the F1 Score 
pred_val = [0,0,0,0,0]

for a in range(0,5):
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    pred_val[a] = f1_score(y_test, y_pred,  average='binary') 
    print ("{} Accuracy: {}".format(names[a],pred_val[a]))


# In[37]:


# Calculate ROC Score

pred_val2=[0,0,0,0,0]

for i in range(0,5):
    train_classifier(clf[i], X_train, y_train)
    y_predx=predict_labels(clf[i], X_test)
    pred_val2[i] = roc_auc_score(y_test,y_predx)
    print ("{} Accuracy: {}".format(names[i],pred_val2[i]))


# In[38]:


# ploating data for F1 Score
y_pos = np.arange(len(objects))
y_val = [ x for x in pred_val]
plt.bar(y_pos,y_val, align='center', alpha=0.8)
plt.xticks(y_pos, objects)
plt.ylabel( 'Score')
plt.title('F1 Score of Models')
plt.show()


# In[39]:


# ploating data for ROC score 
y_pos2=np.arange(len(objects))
y_val2=[i for i in pred_val2]
plt.bar(y_pos2,y_val2,align='center',alpha=0.8)
plt.xticks(y_pos2,objects)
plt.ylabel('Score')
plt.title('ROC Score of Models ')
plt.show()


# In[40]:


# ploating data for Accuracy Score
# ploating data for Accuracy of Models between 1.00 - 0.90 for better visualization
objects = ('','Untunded', 'Tuned','')
y_pos = np.arange(4)
y_val = [0,0.03470790378,0.037062937063,0 ]
plt.bar(y_pos,y_val, align='center',width = 0.5, alpha=0.6)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Naive Bayes')
plt.show()


# In[43]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
best_clf=NaiveBayse
pred = best_clf.predict(X_test)
sns.heatmap(confusion_matrix(y_test, pred), annot = True, fmt = '')


# ## Thanks

# In[ ]:




