#!/usr/bin/env python
# coding: utf-8

# # Importing library

# In[111]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA  # to apply PCA
from sklearn import datasets  # to retrieve the iris Dataset
import pandas as pd  # to load the dataframe
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


# # Reading Dataframe

# In[58]:


review=pd.read_csv("E:\Studies\Programming\Reviews.csv")
review.head()


# In[59]:


review.tail()


# In[60]:


print('The number of entries in the data frame: ', review.shape[0])


# In[61]:


review['ProductId'].nunique()


# In[62]:


review['UserId'].nunique()


# In[63]:


review.isnull().sum()


# In[64]:


review.dropna(inplace=True)


# In[65]:


review.isnull().sum()


# # Data Analysis

# In[66]:


fig = px.histogram(review, x="Score")
fig.update_traces(marker_color="orange",marker_line_color='rgb(8,48,107)',
                  marker_line_width=2.0)
plt.figure(figsize=(5,3))
fig.update_layout(title_text='Product Score')
fig.show()


# from this graph we can say that most of the reviews are postive.score is the ratings given by the customer

# # Word Cloud

# In[67]:


review = review[review['Score'] != 3]
review['sentiment'] = review['Score'].apply(lambda rating : +1 if rating > 3 else -1)
positive = review[review['sentiment'] == 1]
negative = review[review['sentiment'] == -1]
review.head()


# In[68]:


review['sentimentt'] = review['sentiment'].replace({-1 : 'negative'})
review['sentimentt'] = review['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(review, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(10,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[69]:


review.head()


# # Decision Trees

# In[135]:


df=pd.read_csv("E:\Studies\Programming\Reviews.csv")
df.head()


# In[136]:


dt=df
label_encoders={}
categorical_coloums= dt.columns
for columns in categorical_coloums:
    label_encoders[columns]=LabelEncoder()
    dt[columns]=label_encoders[columns].fit_transform(dt[columns])


# In[137]:


X = df.iloc[:,4:5]
y = df.iloc[:,6]


# In[138]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.30, random_state=42)


# In[139]:


classifier = DecisionTreeClassifier(criterion='entropy',random_state=0) 
classifier.fit(X_train,y_train)


# In[140]:


y_pred = classifier.predict(X_test)


# In[141]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[142]:


print(classification_report(y_test, y_pred))


# # Data Cleaning

# In[70]:


review = review.drop(['ProductId','UserId','ProfileName','Id','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary'], axis=1)
review.info(memory_usage='deep')


# In[71]:


review.head()


# # Split the DataFrame

# In[72]:


X_train, X_test, y_train, y_test = train_test_split(review['Text'], review['sentiment'], random_state = 0)
print('X_train first entry: \n\n', X_train[0])
print('\n\nX_train shape: ', X_train.shape)


# # Create a bag of words

# Next, we will use a count vectorizer from the Scikit-learn library.
# This will transform the text in our data frame into a bag of words model, which will contain a sparse matrix of integers. The number of occurrences of each word will be counted and printed.
# We will need to convert the text into a bag-of-words model since the logistic regression algorithm cannot understand text.

# In[73]:


vect = CountVectorizer().fit(X_train)
vect


# In[74]:


# checking the features
feat = vect.get_feature_names()


# In[75]:


len(vect.get_feature_names())


# # Sparse matrix

# In[76]:


X_train_vectorized = vect.transform(X_train)
# the interpretation of the columns can be retreived as follows
# X_train_vectorized.toarray()


# In[77]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[78]:


# accuracy
predictions = model.predict(vect.transform(X_test))


# In[79]:


accuracy_score(y_test, predictions)


# In[80]:


# area under the curve
roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, predictions)


# In[81]:


plt.title('ROC for logistic regression on bag of words', fontsize=20)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.show()


# In[82]:


# coefficient determines the weight of a word (positivity or negativity)
# checking the top 10 positive and negative words

# getting the feature names
feature_names = np.array(vect.get_feature_names())

# argsort: Integer indicies that would sort the index if used as an indexer
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


# # Multinomial Naive Bayes Algorithm

# In[83]:


model=MultinomialNB()
model.fit(X_train_vectorized,y_train)


# In[84]:


predictions=model.predict(vect.transform(X_test))
roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)


# It gives us a score of 0.844 which is good.

# # Bigrams

# In[85]:


vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)


# In[86]:


X_train_vectorized = vect.transform(X_train)


# In[87]:


len(vect.get_feature_names())


# In[88]:


feat = vect.get_feature_names()


# In[89]:


# the number of features has increased again
# checking for the AUC
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# In[90]:


predictions = model.predict(vect.transform(X_test))


# In[91]:


accuracy_score(y_test, predictions)


# In[92]:


roc_auc = roc_auc_score(y_test, predictions)
print('AUC: ', roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, predictions)


# In[93]:


plt.title('ROC for logistic regression on Bigrams', fontsize=20)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.show()


# In[94]:


# check the top 10 features for positive and negative
# reviews again, the AUC has improved
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

# print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
# print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))


# In[95]:


new_review = ['The food is not good, I would never buy them again']
print(model.predict(vect.transform(new_review)))


# In[96]:


new_review = ['One would not be disappointed by the food']
print(model.predict(vect.transform(new_review)))


# # PCA

# In[105]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
dt=review
label_encoders={}
categorical_coloums= dt.columns
for columns in categorical_coloums:
    label_encoders[columns]=LabelEncoder()
    dt[columns]=label_encoders[columns].fit_transform(dt[columns])


# In[106]:


#Check the Co-relation between features without PCA
sns.heatmap(dt.corr())


# In[109]:


#Applying PCA
#Taking no. of Principal Components as 4
pca = PCA(n_components = 3)
pca.fit(dt)
data_pca = pca.transform(dt)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3'])
data_pca.head()


# In[110]:


#Checking Co-relation between features after PCA
sns.heatmap(data_pca.corr())


# In[ ]:




