import csv, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

def readRawDataset(rawDatasetFileName):
    with open(rawDatasetFileName, "r") as f:
        featureNames = []
        X = [] # features
        y = [] # classes
        csvReader = csv.reader(f)
        for rowIndex, row in enumerate(csvReader):
            if rowIndex == 0:
                featureNames = [row[0], row[1], row[2], row[3], row[4], row[5]]
#                 featureNames = [row[0], row[1], row[2], row[4], row[5]]
            else:
                X.append([float(row[0]), float(row[1]), float(row[2]), 
                          float(row[3]), float(row[4]), float(row[5])])
#                 X.append([float(row[0]), float(row[1]), float(row[2]),
#                           float(row[4]), float(row[5])])
                y.append(int(row[7]))

    print(f"Read {rawDatasetFileName}. Total sample count: {len(y)}")
    print("Per-class sample count (alert level 0 to 3):")
    print(perClassSampleCounts(y))
    return (X, y, featureNames)

def undersample(rawDatasetFileName,
                downSampling="RandomUnderSampler",
                randomState=0,
                outputFileName="dataset-undersampled.csv"):
    X, y, featureNames = readRawDataset(rawDatasetFileName)
    if downSampling == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=randomState)
        X, y = sampler.fit_resample(X, y)
    print(f"Undersampling with {downSampling} done.")
    print("Per-class sample count (alert level 0 to 3):")
    print(perClassSampleCounts(y))
    
    csvRows = []
    for i, x in enumerate(X):
        csvRow = x + [y[i]]
        csvRows.append(csvRow)
    
    with open(outputFileName, "w") as f:
        writer = csv.writer(f)
        writer.writerow(featureNames)
        writer.writerows(csvRows)
    print(f"Generated {outputFileName}: {len(csvRows)} rows \n")
    return (X, y, featureNames)

def oversample(rawDatasetFileName,
               overSampling="SMOTE",
               randomState=0,
               removeTestData=False, # or X_test
               outputFileName="dataset-oversampled.csv"):
    X, y, featureNames = readRawDataset(rawDatasetFileName)    

    if overSampling == "SMOTE":
        sampler = SMOTE(random_state=randomState)
        X, y = sampler.fit_resample(X, y)

    if overSampling == "SMOTEENN":
        sampler = SMOTEENN(random_state=randomState)
        X, y = sampler.fit_resample(X, y)
    
    print(f"Oversampling with {overSampling} done. Sample cout: {len(X)}")
    print("Per-class sample count (alert level 0 to 3):")
    print(perClassSampleCounts(y))
    
    if removeTestData is not False:
        X_test = removeTestData
        removalCount=0
        for i, x in enumerate(X):
            if x in X_test:
                del X[i]
                del y[i]
                removalCount += 1
#             for j, test in enumerate(X_test):
#                 if np.array_equal(x, test):
#                     del X[i]
#                     del y[i]
#                     removalCount += 1
        print(f"Test data (X_test) size: {len(X_test)}")
        print(f"Test data (X_test) removed from the oversampled dataset. Removal count: {removalCount} ")

    csvRows = []
    for i, x in enumerate(X):
        csvRow = x + [y[i]]
        csvRows.append(csvRow)
    
    with open(outputFileName, "w") as f:
        writer = csv.writer(f)
        writer.writerow(featureNames)
        writer.writerows(csvRows)
    print("Per-class sample count (alert level 0 to 3):")
    print(perClassSampleCounts(y))
    print(f"Generated {outputFileName}: {len(csvRows)} rows, "\
          f"{len(np.unique(X, axis=0))} unique rows, "\
          f"{len(csvRows) - len(np.unique(X, axis=0))} duplicated rows\n")
    return (X, y, featureNames)

def readData(inputDatasetFileName):
    print(f"Reading {inputDatasetFileName}")
    with open(inputDatasetFileName, "r") as f:
        featureNames = []
        X = [] # features
        y = [] # classes
        csvReader = csv.reader(f)
        for rowIndex, row in enumerate(csvReader):
            if rowIndex == 0:
                featureNames = [row[0], row[1], row[2], row[3], row[4], row[5]]
#                 featureNames = [row[0], row[1], row[2], row[4], row[5]]
            else:
                X.append([float(row[0]), float(row[1]), float(row[2]), 
                          float(row[3]), float(row[4]), float(row[5])])
#                 X.append([float(row[0]), float(row[1]), float(row[2]),
#                           float(row[4]), float(row[5])])
                y.append(int(row[6]))
    print(f"Total sample count: {len(y)}")
    print("Per-class sample count (alert level 0 to 3):")
    print(perClassSampleCounts(y))
    return (X, y, featureNames)    

def perClassSampleCounts(y):
    return [y.count(0), y.count(1), y.count(2), y.count(3)]

def minMaxScaling(X):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print("Min-max scaling done.")
    return X


if __name__ == "__main__":
    rawDatasetFileName = "dataset.csv"
    
    print("***** Undersampling")
    X, y, featureNames = undersample(rawDatasetFileName)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("***** Oversampling with SMOTE")
    oversample(rawDatasetFileName)

    print("***** Oversampling with SMOTE, then removing test data (X_test)")
    oversample(rawDatasetFileName,
               removeTestData = X_test,
               outputFileName = "dataset-oversampled-SMOTE-testdata-removed.csv")

    print("***** Oversampling with SMOTEENN, then removing test data (X_test)")
    oversample(rawDatasetFileName,
               overSampling = "SMOTEENN",
               removeTestData = X_test,
               outputFileName = "dataset-oversampled-SMOTEENN-testdata-removed.csv")

# print( np.array_equal(X_test, X_test2) )
# print(X_test[:5])
# print(X_test2[:5])

