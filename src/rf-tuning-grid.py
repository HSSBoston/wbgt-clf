from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from wbgt_metrics import f1_score_loose, f1_loose_scorer
import numpy as np, csv, time
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from dataset_prep import undersample, oversample 

rawDatasetFileName = "dataset.csv"
X, y, featureNames = undersample(rawDatasetFileName)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

X_train, y_train, featureNames = oversample(rawDatasetFileName,
                                            overSampling="SMOTE", # or SMOTEENN
                                            removeTestData = X_test)
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

parameters = {
#                 "n_estimators": [2000],
#               "criterion": ["gini", "entropy", "log_loss"],
#               "criterion": ["gini"],
#               "max_depth": [None]+[i for i in range(1, 21)],
              "max_depth": list(range(8, 11, 1)),
#               "min_samples_split": [i for i in range(2, 51)],
              "min_samples_split": list(range(2, 21, 1)),
              "min_samples_leaf":  list(range(10, 15, 1)),
#               "max_features": [None, 2, 3, 4, 5],
#               "max_leaf_nodes": list(range(2, 50, 2)),
#               "ccp_alpha": np.arange(0, 0.1, 0.001).tolist(),
              }

clf = RandomForestClassifier(random_state=0)
skf = StratifiedKFold(n_splits=5)

startTime = time.time()

# Default: cv=5, StratifiedKFold
gcv = GridSearchCV(clf, parameters, cv=skf, n_jobs=-1)
gcv.fit(X, y)

optimalModel = gcv.best_estimator_
accuracy = optimalModel.score(X_train, y_train)
print (f"Accuracy in training: {round(accuracy,3)}")
accuracy = optimalModel.score(X_test, y_test)
print (f"Accuracy in testing: {round(accuracy,3)}")

y_predicted = optimalModel.predict(X_test)
f1score = f1_score(y_test, y_predicted, average="macro")
print(f"F1 score: {round(f1score, 3)}")

cm = confusion_matrix(y_test, y_predicted)
print(cm)

f1LooseScore = f1_score_loose(cm)
print(f"F1 loose score: {round(f1LooseScore, 3)}")

print("Best parameters: ", gcv.best_params_)

endTime = time.time()
print(f"{round(endTime-startTime)} sec, {round( (endTime-startTime)/60, 1 )} min")

scores = cross_val_score(optimalModel, X, y, cv=skf, scoring="f1_macro")
print(f"Cross validation F1 score w/ StratifiedKFold: {round(np.mean(scores),3)}")

scores = cross_val_score(optimalModel, X, y, cv=skf, scoring=f1_loose_scorer)
print(f"Cross validation F1 loose score w/ StratifiedKFold: {round(np.mean(scores),3)}")

# 
# sskf = StratifiedShuffleSplit(n_splits=10, test_size=0.3)
# scores = cross_val_score(optimalModel, X, y, cv=sskf)
# print(f"Cross validation score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")


