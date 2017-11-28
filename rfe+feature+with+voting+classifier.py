
# coding: utf-8

# In[18]:

# Load CSV
import pandas as pd
filename = 'fish.csv'
data = pd.read_csv(filename, ',')


# In[19]:

data


# In[20]:

columns = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report']


# In[21]:

X = pd.DataFrame(data, columns=columns)


# In[22]:

X


# In[23]:

y = data['label']


# In[24]:

y


# In[41]:

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
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
from sklearn.svm import LinearSVC
numpy.set_printoptions(threshold=numpy.nan)


# In[42]:

clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=1)
clf3 = RandomForestClassifier()
acc = []
roc = []


# In[43]:

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
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft',weights=[1, 1, 4])
    
    
    scores = cross_val_score(eclf, X1, y, cv=5, scoring='accuracy')
    acc.append(scores.mean())
    print("VotingClassifier Accuracy: %0.2f " % (scores.mean())) 
    scores1 = cross_val_score(eclf, X1, y, cv=5, scoring='roc_auc')
    roc.append(scores1.mean())
    print("VotingClassifier AUC: %0.2f " % (scores1.mean()))  


# In[44]:

acc


# In[45]:

roc


# In[ ]:




# In[ ]:




# In[ ]:



