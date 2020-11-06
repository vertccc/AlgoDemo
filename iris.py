import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)

x_new = np.array([3,5,4,2])
y_new = knn.predict(x_new.reshape(1,-1))
y_new.item()
knn.predict([[1,2,3,4]])

X_new = [[3,5,4,2],[5,4,3,2]]
knn.predict(X_new)



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,y)
logreg.predict(X_new)
logreg.predict_proba(X_new)[0:10,:]
from sklearn.preprocessing import binarize
binarize(y_pred,0.4)

from sklearn import metrics
y_pred = knn.predict(X)
print(metrics.accuracy_score(y,y_pred))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)


sns.pairplot(sns.load_dataset("iris"),hue="species")

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train,y_train)
zip(['a','b','c'],linreg.coef_)
y_pred = linreg.predict(x_test)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=1,shuffle=False)
kf.get_n_splits(X)


# gridsearchcv can automatically do this
from sklearn.model_selection import cross_val_score
for k in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    print(k,score.mean(),score)
from sklearn.model_selection import GridSearchCV
krange = range(1,30)
paras = dict(n_neighbors=krange)
grid = GridSearchCV(knn,paras,cv=10,scoring='accuracy')
grid.fit(X,y)
grid.best_params_
grid.best_estimator_
# use the best estimator
grid.predict([1,4,5,2])


from sklearn.model_selection import RandomizedSearchCV
grid = RandomizedSearchCV(knn,paras,cv=10,n_iter=10,random_state=10)


# ROC, AUC 
# confusion matrix TP TN FP FN (True False Positive Negative)
# False Positive (Type I error)
# False Negatives (Type II error)

from sklearn.metrics import roc_curve,roc_auc_score
# fpr = fp/(fp+tn)  tpr = tp/(tp+fn)
fpr, tpr, thresholds = roc_curve(y_test,y_pred,pos_label=2)
threshold = 0.3
print('sensitivity: {}'.format(tpr[thresholds>threshold][-1]))
print('specificity: {}'.format(1-fpr[thresholds>threshold][-1]))
roc_auc_score(y_test,y_pred)