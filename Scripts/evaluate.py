
import json

with open("output/groundguard_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

TP = 0
TN = 0
FP = 0
FN = 0

for r in results:

    actual = r["is_corrupted"]
    predicted = r["flagged"]

    if actual and predicted:
        TP += 1

    elif not actual and not predicted:
        TN += 1

    elif not actual and predicted:
        FP += 1

    elif actual and not predicted:
        FN += 1


total = len(results)

print("\n========== GROUNGuard EVALUATION ==========")

print(f"Total cases : {total}")
print(f"Corrupted   : {TP + FN}")
print(f"Clean       : {TN + FP}")

print("\nConfusion Matrix")
print("----------------")
print(f"True Positive  (TP): {TP}")
print(f"True Negative  (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")

if total > 0:

    accuracy = (TP + TN) / total

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

else:
    accuracy = 0
    precision = 0
    recall = 0

print("\nMetrics")
print("----------------")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")

