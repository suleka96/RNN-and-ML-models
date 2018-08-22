import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def make_dict():
    direc = "enron/emails/"
    files = os.listdir(direc)
    emails = [direc+email for email in files]

    words = []

    for email in emails:
        f = open(email,encoding="utf8", errors='ignore')
        blob = f.read()
        words += blob.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]

    return dictionary.most_common(3000)

def make_dataset(dictionary):
    direc = "enron/emails/"
    files = os.listdir(direc)
    emails = [direc + email for email in files]
    print(len(emails))

    feature_set = []
    labels = []

    for email in emails:
        data = []
        f = open(email, encoding="utf8", errors='ignore')
        words = f.read().split(' ')

        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)
        if "ham" in email:
            labels.append(0)
        else:
            labels.append(1)
    return feature_set, labels


d = make_dict()
print("1 to 50")
features, labels = make_dataset(d)
print(len(features), "   ", len(labels))

X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=42,  shuffle=False)

# mlnb= MultinomialNB()
# mlnb.fit(X_train,y_train)
# pred = mlnb.predict(X_test)
#
# clf = LogisticRegression()
# clf.fit(X_train, y_train)
# p = clf.predict(X_test)

# param_grid = [{'C': range(1, 50), 'kernel': ['linear']}]
#
# clf = GridSearchCV(SVC(), param_grid, cv=5)
# clf.fit(X_train, y_train)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()

# svctest = SVC(kernel='linear', random_state=42, C=clf.best_params_["C"])
# svctest.fit(X_train, y_train)
# prediction = svctest.predict(X_test)

svctest2 = SVC(kernel='linear', random_state=42, C=1)
svctest2.fit(X_train, y_train)
prediction2 = svctest2.predict(X_test)



# rcf = RandomForestClassifier( random_state=42)
# rcf.fit(X_train, y_train)
# predRFC = rcf.predict(X_test)


# print("----------------------Logistic Regression------------------------------")
# print("Accuracy score")
# print(accuracy_score(y_test, p))
# print("F1 score")
# print(f1_score(y_test, p, average='macro'))
# print("Recall")
# print(recall_score(y_test, p, average='macro'))
# print("Precision")
# print(precision_score(y_test, p, average='macro'))
#
# print("----------------------MultinomialNB------------------------------")
# print("Accuracy score")
# print(accuracy_score(y_test, pred))
# print("F1 score")
# print(f1_score(y_test, pred, average='macro'))
# print("Recall")
# print(recall_score(y_test, pred, average='macro'))
# print("Precision")
# print(precision_score(y_test, pred, average='macro'))

# print("----------------------SVC------------------------------")
# print("Accuracy score")
# print(accuracy_score(y_test, prediction))
# print("F1 score")
# print(f1_score(y_test, prediction, average='macro'))
# print("Recall")
# print(recall_score(y_test, prediction, average='macro'))
# print("Precision")
# print(precision_score(y_test, prediction, average='macro'))
# print(classification_report(y_test, prediction))
#
# print()
# print()

print("Accuracy score")
print(accuracy_score(y_test, prediction2))
print("F1 score")
print(f1_score(y_test, prediction2, average='macro'))
print("Recall")
print(recall_score(y_test, prediction2, average='macro'))
print("Precision")
print(precision_score(y_test, prediction2, average='macro'))
print(classification_report(y_test, prediction2))


# print("-------------------------Random Forest Classifier-------------------------------------------")
# print("Accuracy score")
# print(accuracy_score(y_test,predRFC))
# print("F1 score")
# print(f1_score(y_test, predRFC, average='macro'))
# print("Recall")
# print(recall_score(y_test, predRFC, average='macro'))
# print("Precision")
# print(precision_score(y_test, predRFC, average='macro'))


# ----------------------Logistic Regression------------------------------
# Accuracy score
# 0.9681159420289855
# F1 score
# 0.9619726354394329
# Recall
# 0.9659443072086233
# Precision
# 0.9582387342420735
# ----------------------MultinomialNB------------------------------
# Accuracy score
# 0.9342995169082126
# F1 score
# 0.9221411945872753
# Recall
# 0.9286099258926566
# Precision
# 0.9163721563391711













