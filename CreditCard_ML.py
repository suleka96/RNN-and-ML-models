
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


#A macro-average will compute the metric independently for each class
#  and then take the average

#a micro-average will aggregate the contributions of all classes to compute
# the average metric.


data = pd.read_csv('creditcard.csv', skiprows=[0], header=None)

X = data.iloc[:, 0:30]
y = data.iloc[:, -1]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle= False)

logreg= LogisticRegression()

logreg.fit(X_train, y_train)
predLogreg = logreg.predict(X_test)

print("-------------------------LOGISTIC REGRESSION-------------------------------------------")
print("Accuracy")
print(accuracy_score(y_test,predLogreg))
print("F1 score")
print(f1_score(y_test, predLogreg, average='macro'))
print("Recall")
print(recall_score(y_test, predLogreg, average='macro'))

# svc = svm.SVC()
# svc.fit(X_train, y_train)
# predSvc = svc.predict(X_test)
#
# print("-------------------------SVM-------------------------------------------")
# print("Accuracy")
# print(accuracy_score(y_test,predSvc))
# print("F1 score")
# print(f1_score(y_test, predSvc, average='macro'))

rcf = RandomForestClassifier( random_state=0)
rcf.fit(X_train, y_train)
predRFC = rcf.predict(X_test)

print("-------------------------Random Forest Classifier-------------------------------------------")
print("Accuracy")
print(accuracy_score(y_test,predRFC))
print("F1 score")
print(f1_score(y_test, predRFC, average='macro'))
print("Recall")
print(recall_score(y_test, predRFC, average='macro'))





