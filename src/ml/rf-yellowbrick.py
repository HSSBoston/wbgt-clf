from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt, numpy as np
from dataset_prep import undersample, oversample 
from yellowbrick.model_selection import validation_curve
from matplotlib.ticker import MaxNLocator

rawDatasetFileName = "dataset.csv"
X, y, featureNames = undersample(rawDatasetFileName)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

X_train, y_train, featureNames = oversample(rawDatasetFileName,
                                            overSampling="SMOTE", # or SMOTEENN
                                            removeTestData = X_test)
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

skf = StratifiedKFold(n_splits=5)

# fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1, 20, 1))

visualizer = validation_curve(
    RandomForestClassifier(random_state=0),
    X, y,
    param_name = "n_estimators", param_range = range(1, 500, 50),
#     param_name = "max_depth", param_range = range(1, 21),
#     param_name = "max_leaf_nodes", param_range = range(2, 201),
#     param_name = "min_samples_split", param_range = range(2, 51, 2), #def 2
#     param_name = "min_samples_leaf", param_range = range(1, 51, 2), #def 1
#     param_name = "max_features", param_range = [1, 2, 3, 4, 5], #def 5
    cv=skf, scoring = "f1_macro",
    ax=ax, n_jobs=-1)

