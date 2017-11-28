from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

filename = 'fish.csv'
data = pd.read_csv(filename)
n_classes = 2

#columns = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report']
#columns = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report']
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

clf = svm.SVC(kernel = 'poly', probability=True)
data1 = pd.DataFrame(data,columns = columns)

for i in range(1,len(columns)):
    #X = pd.DataFrame(data, columns=columns)
    #print("SVM Score for Feature", i )
    #X = data[[i]]
    X = data1.iloc[:,0:i]
    #print(X)
    y = data['label']
    x_train_original,x_test_original,y_train_original,y_test_original=train_test_split(X,y,test_size=0.25,random_state = 0)
    clf.fit(x_train_original,y_train_original)
    predictions=clf.predict(x_test_original)
    print("", accuracy_score(y_test_original, predictions))
    #print("", roc_auc_score(y_test_original, predictions))
    #print(roc_auc_score(y_test_original, predictions))
    print(",")

y_score = clf.predict_proba(x_test_original)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_original, y_score)
roc_auc = auc(false_positive_rate, true_positive_rate)

'''
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


y_score = clf.fit(x_train_original, y_train_original).decision_function(x_test_original)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_original, predictions)
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 2
plt.figure()
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
 '''
