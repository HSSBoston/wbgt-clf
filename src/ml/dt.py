from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from wbgt_metrics import f1_score_loose, f1_loose_scorer
import numpy as np, matplotlib.pyplot as plt, time, pickle, joblib
from dataset_prep import undersample, oversample 

rawDatasetFileName = "dataset.csv"
X, y, featureNames = undersample(rawDatasetFileName)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

# Training data
X_train, y_train, featureNames = oversample(rawDatasetFileName,
                                            overSampling="SMOTE", # or SMOTEENN
                                            removeTestData = X_test)
# Cross validation data
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

dTree = DecisionTreeClassifier(random_state=0)

startTime = time.time()
dTree.fit(X_train, y_train)
endTime = time.time()
print(f"Training time (s): {round(endTime-startTime, 2)}")

y_predicted = dTree.predict(X_train)
f1score = f1_score(y_train, y_predicted, average="macro")
print(f"Training accuracy (%) in F1: {round(f1score, 3)}")

startTime = time.time()
y_predicted = dTree.predict(X_test)
endTime = time.time()
print(f"Classification time (s): {round(endTime-startTime, 3)}")
print(f"Classification time per sample (s): {round((endTime-startTime)/len(X_test), 4)}")

f1score = f1_score(y_test, y_predicted, average="macro")
print(f"Testing accuracy (%) in F1: {round(f1score, 3)}")

cm = confusion_matrix(y_test, y_predicted)
print(cm)

f1LooseScore = f1_score_loose(cm)
print(f"F1 loose score: {round(f1LooseScore, 3)}")

# K-fold cross validation
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(dTree, X, y, cv=skf, scoring="f1_macro")
print(f"Cross validation F1 score w/ StratifiedKFold: {round(np.mean(scores),3)}")

# sskf = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
# scores = cross_val_score(dTree, X, y, cv=sskf, scoring="f1_macro")
# print(f"Cross validation F1 score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")
# 
# scores = cross_val_score(dTree, X, y, cv=skf, scoring=f1_loose_scorer)
# print(f"Cross validation F1 loose score w/ StratifiedKFold: {round(np.mean(scores),3)}")
# 
# scores = cross_val_score(dTree, X, y, cv=sskf, scoring=f1_loose_scorer)
# print(f"Cross validation F1 loose score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")

print(dTree.feature_importances_)

pImportance = permutation_importance(dTree, X, y, n_repeats=100, random_state=0)
print(pImportance["importances_mean"])

# Save the model as a file
pickle.dump(dTree, open("dt.pkl", "wb"))
# dTree = pickle.load(open("dt.pkl", "rb"))

joblib.dump(dTree, "dt.joblib", compress=3)
# dTree = joblib.load("dt.joblib")


