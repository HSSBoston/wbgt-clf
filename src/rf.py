from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from wbgt_metrics import f1_score_loose, f1_loose_scorer
import numpy as np, csv, matplotlib.pyplot as plt, time, pickle
from sklearn.tree import plot_tree
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

# clf = RandomForestClassifier(random_state=0)
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

startTime = time.time()
clf.fit(X_train, y_train)
endTime = time.time()
print(f"Training time (s): {round(endTime-startTime, 2)}")


# accuracy = clf.score(X_train, y_train)
# print (f"Accuracy in training: {round(accuracy,3)}")
# accuracy = clf.score(X_test, y_test)
# print (f"Accuracy in testing: {round(accuracy,3)}")

y_predicted = clf.predict(X_train)
f1score = f1_score(y_train, y_predicted, average="macro")
print(f"Training accuracy (%) in F1: {round(f1score, 3)}")

startTime = time.time()
y_predicted = clf.predict(X_test)
endTime = time.time()
print(f"Classification time (s): {round(endTime-startTime, 3)}")
print(f"Classification time per sample (s): {round((endTime-startTime)/len(X_test), 4)}")

f1score = f1_score(y_test, y_predicted, average="macro")
print(f"Testing accuracy (%) in F1: {round(f1score, 3)}")

cm = confusion_matrix(y_test, y_predicted)
print(cm)

f1LooseScore = f1_score_loose(cm)
print(f"F1 loose score: {round(f1LooseScore, 3)}")

skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
print(f"Cross validation F1 score w/ StratifiedKFold: {round(np.mean(scores),3)}")

# sskf = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
# scores = cross_val_score(clf, X, y, cv=sskf, scoring="f1_macro")
# print(f"Cross validation F1 score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")

# scores = cross_val_score(clf, X, y, cv=skf, scoring=f1_loose_scorer)
# print(f"Cross validation F1 loose score w/ StratifiedKFold: {round(np.mean(scores),3)}")

# sskf = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
# scores = cross_val_score(clf, X, y, cv=sskf, scoring=f1_loose_scorer)
# print(f"Cross validation F1 loose score w/ StratifiedShuffleSplit: {round(np.mean(scores),3)}")

print(clf.feature_importances_)
# pImportance = permutation_importance(clf, X, y, n_repeats=100, random_state=0)
# print(pImportance["importances_mean"])

# Save the model as a file
pickle.dump(clf, open("rf.pkl", "wb"))
# dTree = pickle.load(open("dt.pkl", "rb"))



# iris = load_iris()
# # print(type(iris))
# X = iris.data   # numpy.ndarray, features
# y = iris.target # numpy.ndarray, species
# print (X[0:5,:])
#     # Sepal Length, Sepal Width, Petal Length, Petal Width
# print(y)
#     # 0: Setosa
#     # 1: Versicolor
#     # 2: Virginica
# print(f"Feature names: {iris.feature_names}")
# print(f"Class names: {iris.target_names}")
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#     # 30% for testing, 70% for training
#     # Deterministic (non-random) sampling
# 
# clf = tree.DecisionTreeClassifier(max_depth=3)
#     # Too shallow tree: poorer classification
#     # Too deep: overfitting
# clf.fit(X_train, y_train)
# print ("Accuracy:", clf.score(X_test, y_test))
# 
# print(f"Correct result: {y_test}")
# print(f"Predicted:      {clf.predict(X_test)}")
# 
# # K分割交差検証
# stratifiedkfold = StratifiedKFold(n_splits=10)  #K=10分割
# scores = cross_val_score(clf, X, y, cv=stratifiedkfold)
# # print(f"Cross-Validation scores: {scores}")   # 各分割におけるスコア
# print(f"Cross validation score: {np.mean(scores)}")  # スコアの平均値
# 
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
