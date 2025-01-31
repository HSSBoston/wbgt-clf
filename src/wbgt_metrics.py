from sklearn.metrics import confusion_matrix

def f1_loose_scorer(estimator, X, y):
    y_predicted = estimator.predict(X)
    cm = confusion_matrix(y, y_predicted)
    return f1_score_loose(cm)   
    
def f1_score_loose(confusionMatrix):
    classCount = len(confusionMatrix[0])
    f1Vals = []
    
    for classIndex in range(classCount):
        p = precision(confusionMatrix, classIndex)
        r = recall(confusionMatrix, classIndex)
        f1 = (2 * p * r)/(p + r)
#         print(f"***** {f1}")
        f1Vals.append(f1)
#     print(f1Vals)
    return sum(f1Vals)/classCount

def precision(confusionMatrix, classIndex):    
    column = [confusionMatrix[0][classIndex],
              confusionMatrix[1][classIndex],
              confusionMatrix[2][classIndex],
              confusionMatrix[3][classIndex]]
#     print("Column", column)

    if classIndex == 0:
        tpList = column[0 : classIndex+1]
    else:
        tpList = column[classIndex-1 : classIndex+1]
    tp = sum(tpList)
#     print("TP", tpList, tp)
    
    fpList = column[classIndex+1:]
    fp = sum(fpList)
#     print("FP", fpList, fp)
    
    precision = tp/(tp+fp)
#     print("Precision", precision)
    return precision

def recall(confusionMatrix, classIndex):
    row = confusionMatrix[classIndex]
#     print("Row", row)
    
    tpList = row[classIndex : classIndex+2]
    tp = sum(tpList)
#     print("TP", tpList, tp)

    fnList = row[:classIndex]
    fn = sum(fnList)
#     print("FN", fnList, fn)

    recall = tp/(tp+fn)
#     print("Recall", recall)
    return recall
    
#     for classIndex in classCount:
        
        

if __name__ == "__main__":
    confusionMatrix = [[148,   9,   0,   0],
                       [ 14, 129,  21,   0],
                       [  0,  18, 137,   8],
                       [  0,   4,  28, 136]]
    print(f1_score_loose(confusionMatrix))
    

#     for i in range(4):
#         precision(confusionMatrix, i)
#     for i in range(4):
#         recall(confusionMatrix, i)
