from sklearn.model_selection import train_test_split
from dataset_prep import readRawDataset, undersample, oversample 
import joblib, numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import dtreeviz

clf = joblib.load("dt.joblib")

rawDatasetFileName = "dataset.csv"
X, y, featureNames = readRawDataset(rawDatasetFileName)

plot_tree(clf,
          feature_names = featureNames,
          class_names = ["0", "1", "2", "3"],
          fontsize=10,
          max_depth=5,
          filled=True)
plt.show()


# rawDatasetFileName = "dataset.csv"
# X, y, featureNames = undersample(rawDatasetFileName)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.2, random_state=0)
# 
# # Training data
# X_train, y_train, featureNames = oversample(rawDatasetFileName,
#                                             overSampling="SMOTE", # or SMOTEENN
#                                             removeTestData = X_test)
# # Cross validation data
# X = np.concatenate([X_train, X_test])
# y = np.concatenate([y_train, y_test])
# 
# viz_model = dtreeviz.model(
#                 clf,
#                 X_train=X_test,
#                 y_train=y_test,
#                 target_name='variety',
#                 feature_names=featureNames,
#                 class_names = ["0", "1", "2", "3"])
# viz_model.view(scale=0.8)


# if __name__ == "__main__":
    