
# coding: utf-8

# In[38]:

# Load CSV
import pandas as pd
filename = 'fish.csv'
data = pd.read_csv(filename, ',')


# In[39]:

data


# In[40]:

columns = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report']


# In[41]:

X = pd.DataFrame(data, columns=columns)


# In[42]:

X


# In[43]:

y = data['label']


# In[44]:

y


# In[45]:

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


# In[46]:


acc = []
roc = []


# In[47]:

for i in range(1,31):
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

    X1 = pd.DataFrame(data, columns=new_features)
    print("SVM Score for Features", i )
    #clf = AdaBoostClassifier(n_estimators=8)
    clf = svm.SVC()
    scores = cross_val_score(clf,X1,y, scoring='accuracy')
    acc.append(scores.mean())
    print("AdaBoostClassifier Accuracy: %0.2f " % (scores.mean())) 
    scores1 = cross_val_score(clf, X1, y, cv=5, scoring='roc_auc')
    roc.append(scores1.mean())
    print("AdaBoostClassifier AUC: %0.2f " % (scores1.mean()))  


# In[48]:

acc


# In[49]:

roc


# In[ ]:




# In[ ]:




# In[ ]:



