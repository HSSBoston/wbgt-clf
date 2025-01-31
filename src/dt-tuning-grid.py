from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, confusion_matrix
from wbgt_metrics import f1_score_loose, f1_loose_scorer
import numpy as np, csv, time
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# import dtreeviz
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
#               "criterion": ["gini", "entropy", "log_loss"],
#               "max_depth": [None]+[i for i in range(1, 21)],
              "max_depth": list(range(9, 11, 1)),
              "min_samples_split": list(range(40, 45, 1)),
              "min_samples_leaf":  list(range(20, 30, 1)),
#               "max_features": [None, "sqrt"],
#               "max_leaf_nodes": list(range(25, 77, 2)),
#               "ccp_alpha": np.arange(0, 0.1, 0.001).tolist(),
              }

dTree = DecisionTreeClassifier(random_state=0)
skf = StratifiedKFold(n_splits=5)
sskf = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

startTime = time.time()

# Default: cv=5, StratifiedKFold
gcv = GridSearchCV(dTree, parameters, cv=skf, n_jobs=-1)
gcv.fit(X_train, y_train)
# gcv.fit(X, y)

optimalModel = gcv.best_estimator_
optimalModel.fit(X_train, y_train)
accuracy = optimalModel.score(X_train, y_train)
print (f"Accuracy in training: {round(accuracy,3)}")
accuracy = optimalModel.score(X_test, y_test)
print (f"Accuracy in testing: {round(accuracy,3)}")

print("Best parameters: ", gcv.best_params_)

scores = cross_val_score(optimalModel, X, y, cv=skf, scoring="f1_macro")
print(f"Cross validation F1 score w/ StratifiedKFold: {round(np.mean(scores),3)}")

scores = cross_val_score(dTree, X, y, cv=skf, scoring=f1_loose_scorer)
print(f"Cross validation F1 loose score w/ StratifiedKFold: {round(np.mean(scores),3)}")

# scores = cross_val_score(optimalModel, X, y, cv=sskf)
# print(f"Cross validation score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")

endTime = time.time()

print(optimalModel.feature_importances_)
pImportance = permutation_importance(optimalModel, X, y, n_repeats=100, random_state=0)
print(pImportance["importances_mean"])

print(f"Exec time: {round(endTime-startTime)} sec, {round((endTime-startTime)/60, 1)} min")








# print( clf.feature_importances_)
# 
# plot_tree(clf,
#           feature_names=iris.feature_names,
#           class_names=iris.target_names,
#           fontsize=10,
#           filled=True)
# plt.show()
# 
# # viz_model = dtreeviz.model(clf,
# #                X_train=X_train,
# #                y_train=y_train,
# #                target_name='Class',
# #                feature_names=iris.feature_names,
# #                class_names=iris.target_names)
# # viz_model.view(scale=0.8)
