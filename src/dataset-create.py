# This program reads raw CSV files in the "csv-files" folder and generates
# a single combined CSV file. It determines the alert level for each row
# (each data sample): 0 to 3. It also remove duplicated rows (samples).

import os, csv
import numpy as np

outputFileName = "dataset.csv"

csvFileNames = os.listdir("csv-files")
if os.path.isfile("csv-files/.DS_Store"):
    csvFileNames.remove(".DS_Store")
print(f"Importing: {csvFileNames}")

alertColors = ["Green", "Yellow", "Orange", "Red", "Black"]
csvHeader = []
csvRows = []

for inputFileName in csvFileNames:
    with open("csv-files/" + inputFileName, "r") as f:
        csvReader = csv.reader(f)
        for rowIndex, row in enumerate(csvReader):
            if rowIndex == 0:
                csvHeader = [row[3], row[5], row[6], row[7], row[8], row[10], row[9],
                             "class", "alert color"]
            else:
                if "" in row: continue
                wbgtF = float(row[9])
#                 if wbgtF > 86.2:   classNum = 4
#                 elif wbgtF > 84.2: classNum = 3
#                 elif wbgtF > 81.1: classNum = 2
#                 elif wbgtF > 76.3: classNum = 1
#                 else:              classNum = 0
                
#                 if   wbgtF >= 86: classNum = 4
#                 elif wbgtF >= 84: classNum = 3
#                 elif wbgtF >= 81: classNum = 2
#                 elif wbgtF >= 76: classNum = 1
#                 else:             classNum = 0

                if   wbgtF >= 86: classNum = 3
                elif wbgtF >= 84: classNum = 2
                elif wbgtF >= 81: classNum = 1
                else:             classNum = 0

                newRow = [row[3], row[5], row[6], row[7], row[8], row[10], row[9],
                          classNum, alertColors[classNum]]
                csvRows.append(newRow)
    print("Finished reading " + inputFileName)

print("CSV row count:", len(csvRows))
print("CSV unique row count:", len(np.unique(csvRows, axis=0)))

# Remove duplicated rows. Keys in a set has to be immutable (tuples).
# {(...), (...), ...}
uniqueCsvRowsSet = set([tuple(row) for row in csvRows])
# Unique rows are placed in a list of lists: [[...], [...], ...]
uniqueCsvRows = [list(row) for row in uniqueCsvRowsSet]
print(f"Duplicated rows removed. Before: {len(csvRows)}, After: {len(uniqueCsvRows)}")

with open(outputFileName, "w") as f:
    writer = csv.writer(f)
    writer.writerow(csvHeader)
    writer.writerows(uniqueCsvRows)
print(f"Generated {outputFileName}: {len(uniqueCsvRows)} rows")

# alertZeroCount = alertOneCount = alertTwoCount = alertThreeCount = alertFourCount = 0
alertZeroCount = alertOneCount = alertTwoCount = alertThreeCount = 0

with open(outputFileName, "r") as f:
    csvReader = csv.reader(f)
    for rowIndex, row in enumerate(csvReader):
        if rowIndex == 0:
            continue
        else:
            if   row[7] == "0": alertZeroCount  += 1
            elif row[7] == "1": alertOneCount   += 1
            elif row[7] == "2": alertTwoCount   += 1
            elif row[7] == "3": alertThreeCount += 1
#             elif row[7] == "4": alertFourCount  += 1

# print("Sample counts (alert level 0 to 4:")
# print(f"  {alertZeroCount}, {alertOneCount}, {alertTwoCount}, {alertThreeCount}, {alertFourCount}")

print("Sample counts (alert level 0 to 3):")
print(f"  {alertZeroCount}, {alertOneCount}, {alertTwoCount}, {alertThreeCount}")

