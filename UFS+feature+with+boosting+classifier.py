
# coding: utf-8

# In[11]:

# Load CSV
import pandas as pd
filename = 'fish.csv'
data = pd.read_csv(filename, ',')


# In[12]:

data


# In[13]:

columns = ['SSLfinal_State',
'URL_of_Anchor',
'Request_URL',
'Prefix_Suffix',
'Domain_registeration_length',
'Links_in_tags',
'web_traffic',
'having_Sub_Domain',
'age_of_domain',
'Google_Index',
'having_IP_Address',
'Statistical_report',
'Abnormal_URL',
'DNSRecord',
'double_slash_redirecting',
'Favicon',
'having_At_Symbol',
'HTTPS_token',
'Iframe',
'Links_pointing_to_page',
'on_mouseover',
'Page_Rank',
'popUpWidnow',
'port',
'Redirect',
'RightClick',
'SFH',
'Shortining_Service',
'Submitting_to_email','URL_Length']


# In[14]:

X = pd.DataFrame(data, columns=columns)


# In[15]:

X


# In[16]:

y = data['label']


# In[17]:

y


# In[18]:

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy 
from sklearn.svm import LinearSVC
numpy.set_printoptions(threshold=numpy.nan)
from sklearn import svm


# In[19]:


acc = []
roc = []


# In[20]:

for i in range(1,len(columns)):
    estimator = LinearSVC()
    selector = RFE(estimator, i , step=1)
    selector = selector.fit(X, y)
    #print(selector.support_)
    #print(selector.ranking_)
    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features

    for bool, feature in zip(mask, columns):
        if bool:
            new_features.append(feature)

    X1 = X.iloc[:,0:i]
    print("SVM Score for Features", i )
    clf = AdaBoostClassifier(n_estimators=8)
   
    scores = cross_val_score(clf,X1,y, scoring='accuracy')
    acc.append(scores.mean())
    print("AdaBoostClassifier Accuracy: %0.2f " % (scores.mean())) 
    scores1 = cross_val_score(clf, X1, y, cv=5, scoring='roc_auc')
    roc.append(scores1.mean())
    print("AdaBoostClassifier AUC: %0.2f " % (scores1.mean()))  


# In[21]:

acc


# In[22]:

roc


# In[ ]:




# In[ ]:




# In[ ]:



