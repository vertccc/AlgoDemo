import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# corpus = ['This is the first document.',
#     'This document is the second document.',
#      'And this is the third one.',
#      'Is this the first document?',]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(X.toarray())
# df = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
# train = vectorizer.transform(["this is"]).toarray()
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_csv(url,header=None,names=['label','message'],sep='\t')
sms.label.value_counts()
sms['label_num'] = sms.label.map({'ham':0,'spam':1})

X = sms.message
y = sms.label_num
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

vect = CountVectorizer()
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)

# logreg and any other classification models
nb = MultinomialNB()
nb.fit(x_train_dtm,y_train)

y_pred_class = nb.predict(x_test_dtm)


metrics.accuracy_score(y_test,y_pred_class)
metrics.confusion_matrix(y_test,y_pred_class)
# column 0 for prob 0, column 1 for prob 1
y_pred_prob = nb.predict_proba(x_test_dtm)[:,1]
metrics.roc_auc_score(y_test,y_pred_prob)


x_train_tokens = vect.get_feature_names()
nb.feature_count_

ham_token_count =  nb.feature_count_[0,:]
spam_token_count =  nb.feature_count_[1,:]

tokens = pd.DataFrame({'token':x_train_tokens,'ham':ham_token_count,'spam':spam_token_count},index='token')


tokens.sample(5,random_state=10)

nb.class_count_

tokens['ham'] = (tokens.ham +  1)/nb.class_count_[0,:]
tokens['spam'] = (tokens.spam + 1)/nb.class_count_[1,:]
tokens['spam_ratio'] = tokens.spam / tokens.ham

tokens.loc['dating','spam_ratio']