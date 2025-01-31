import joblib

exampleFeatures = [14, 38, 73, 0.02, 89, 5]

clf = joblib.load("rf.joblib")
y_predicted = clf.predict([exampleFeatures])
alertLevel = int(y_predicted[0])

print(f"Predicted Heat Alert Level: {alertLevel}")

