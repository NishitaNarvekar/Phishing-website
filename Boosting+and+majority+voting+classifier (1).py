
# coding: utf-8

# In[1]:

# Load CSV
import pandas as pd
filename = 'fish.csv'
data = pd.read_csv(filename, ',')


# In[2]:

data


# In[3]:

columns = ['Prefix_Suffix', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags', 'SFH']


# In[4]:

X = pd.DataFrame(data, columns=columns)


# In[5]:

X


# In[6]:

y = data['label']


# In[7]:

for n,i in enumerate(y):
    if i== -1:
        y[n]=0
y


# In[26]:

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# In[49]:

clf = AdaBoostClassifier(n_estimators=8)


# In[50]:

scores = cross_val_score(clf,X,y)
scores.mean() 


# In[50]:

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy 
numpy.set_printoptions(threshold=numpy.nan)


# In[51]:

clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=1)
clf3 = RandomForestClassifier()


# In[52]:

eclf = VotingClassifier(estimators=[('gnb', clf1), ('dt', clf2), ('rf', clf3)], voting='hard')


# In[53]:

for clf, label in zip([clf1, clf2, clf3, eclf], ['naive Bayes', 'Decision Tree Classifier', 'Random Forest', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))  


# In[ ]:

from sklearn.model_selection import train_test_split
x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(X,y,test_size=0.25,random_state = 0)
eclf.fit(x_train_original,y_train_original)
predictions=clf.predict(x_test_original)


# In[ ]:

predictions


# In[ ]:

y_test_original


# In[ ]:

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='soft',
                        weights=[1, 1, 4])
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Decision Tree Classifier', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))  


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



