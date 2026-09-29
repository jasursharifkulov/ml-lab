import math
import json

# ==========================================
# ДАННЫЕ ИЗ ЗАДАНИЯ
# ==========================================
train_data = [
    {"Traffic": "Low", "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Traffic": "Medium", "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Traffic": "Low", "Weather": "Clear", "Courier Load": "Light", "Distance": "Long", "Target": "ON_TIME"},
    {"Traffic": "Low", "Weather": "Rain", "Courier Load": "Light", "Distance": "Short", "Target": "ON_TIME"},
    {"Traffic": "Medium", "Weather": "Clear", "Courier Load": "Light", "Distance": "Long", "Target": "ON_TIME"},
    {"Traffic": "Medium", "Weather": "Rain", "Courier Load": "Light", "Distance": "Short", "Target": "DELAYED"},
    {"Traffic": "Medium", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Long", "Target": "DELAYED"},
    {"Traffic": "High", "Weather": "Clear", "Courier Load": "Light", "Distance": "Long", "Target": "DELAYED"},
    {"Traffic": "Low", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Long", "Target": "DELAYED"},
    {"Traffic": "High", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Target": "DELAYED"},
    {"Traffic": "High", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Long", "Target": "SEVERE_DELAYED"},
    {"Traffic": "High", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Short", "Target": "SEVERE_DELAYED"},
    {"Traffic": "Medium", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Long", "Target": "SEVERE_DELAYED"},
    {"Traffic": "High", "Weather": "Rain", "Courier Load": "Light", "Distance": "Long", "Target": "SEVERE_DELAYED"},
    {"Traffic": "High", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Long", "Target": "SEVERE_DELAYED"}
]

test_parcels = [
    {"ID": "T1", "Traffic": "Low", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Actual": "ON_TIME"},
    {"ID": "T2", "Traffic": "High", "Weather": "Rain", "Courier Load": "Light", "Distance": "Short", "Actual": "DELAYED"},
    {"ID": "T3", "Traffic": "Medium", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Short", "Actual": "DELAYED"},
    {"ID": "T4", "Traffic": "High", "Weather": "Clear", "Courier Load": "Light", "Distance": "Short", "Actual": "DELAYED"},
    {"ID": "T5", "Traffic": "Medium", "Weather": "Clear", "Courier Load": "Heavy", "Distance": "Short", "Actual": "DELAYED"},
    {"ID": "T6", "Traffic": "High", "Weather": "Rain", "Courier Load": "Heavy", "Distance": "Long", "Actual": "SEVERE_DELAYED"}
]

features = ["Traffic", "Weather", "Courier Load", "Distance"]
classes = ["ON_TIME", "DELAYED", "SEVERE_DELAYED"]

# ==========================================
# TASK 1: ENTROPY AND INFORMATION GAIN LOGIC
# ==========================================
def calculate_entropy(dataset):
    total = len(dataset)
    if total == 0: return 0
    counts = {}
    for row in dataset:
        counts[row["Target"]] = counts.get(row["Target"], 0) + 1
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def split_dataset(dataset, feature):
    subsets = {}
    for row in dataset:
        val = row[feature]
        subsets.setdefault(val, []).append(row)
    return subsets

def calculate_ig(dataset, feature):
    h_d = calculate_entropy(dataset)
    subsets = split_dataset(dataset, feature)
    total = len(dataset)
    h_d_a = sum((len(sub)/total) * calculate_entropy(sub) for sub in subsets.values())
    return h_d_a, h_d - h_d_a

# ==========================================
# TASK 2: SELECT ROOT FEATURE
# ==========================================
print("--- TASK 1 & 2: Root Selection & IG Check ---")
print(f"Programmatic IG(D, Weather): {calculate_ig(train_data, 'Weather')[1]:.5f} (Matches hand calculation 0.19186)")
print("\nFeature            Information Gain")
best_feature = None
max_ig = -1

# Детерминированная сортировка для разрешения ничьих (алфавитный порядок признаков)
for f in sorted(features):
    _, ig = calculate_ig(train_data, f)
    print(f"{f:<18} {ig:.5f}")
    if ig > max_ig:
        max_ig = ig
        best_feature = f

print(f"\nSelected root feature: {best_feature}\n")

# ==========================================
# TASK 3: RECURSIVE ID3 TREE BUILDING
# ==========================================
def get_majority_class(dataset):
    counts = {}
    for row in dataset:
        counts[row["Target"]] = counts.get(row["Target"], 0) + 1
    return max(counts, key=counts.get)

# Сохраняем все уникальные значения для каждого признака изначально
global_feature_values = {f: list(set(row[f] for row in train_data)) for f in features}

def build_id3(dataset, available_features):
    if not dataset: return "DELAYED" # На случай пустоты
    targets = [row["Target"] for row in dataset]
    
    # 1. Stop if all samples belong to one class
    if len(set(targets)) == 1:
        return targets[0]
        
    # 2. If no unused features remain, create a leaf with majority class
    if not available_features:
        return get_majority_class(dataset)
        
    # 3. Choose A* using Information Gain (с алфавитной сортировкой для предсказуемости)
    a_star = max(sorted(available_features), key=lambda f: calculate_ig(dataset, f)[1])
    
    tree = {"feature": a_star, "branches": {}}
    remaining_features = [f for f in available_features if f != a_star]
    
    # 4. Create one branch for every v in V(A*)
    for v in global_feature_values[a_star]:
        # 6. Child subset Dv
        dv = [row for row in dataset if row[a_star] == v]
        if not dv:
            tree["branches"][v] = get_majority_class(dataset)
        else:
            # 5. Remove A* from candidate features for that branch and recurse
            tree["branches"][v] = build_id3(dv, remaining_features)
            
    return tree

tree_model = build_id3(train_data, features)
print("--- TASK 3: Learned ID3 Tree ---")
print(json.dumps(tree_model, indent=4, ensure_ascii=False))
print("\n")

# ==========================================
# TASK 4: PREDICTION WITH PATH LOGGING
# ==========================================
def predict(tree, sample, path_steps=None):
    if path_steps is None:
        path_steps = []
        
    if not isinstance(tree, dict):
        return tree, " -> ".join(path_steps) + f" -> [{tree}]"
        
    feature = tree["feature"]
    val = sample[feature]
    path_steps.append(f"{feature}={val}")
    
    if val in tree["branches"]:
        return predict(tree["branches"][val], sample, path_steps)
    else:
        # Резервный вариант, если встретится неведомое значение
        return "DELAYED", " -> ".join(path_steps) + " -> [Fallback: DELAYED]"

print("--- TASK 4: Test Parcel Predictions ---")
predictions = []
for t in test_parcels:
    pred, path = predict(tree_model, t, path_steps=[])
    predictions.append((t["ID"], t["Actual"], pred))
    print(f"Parcel {t['ID']}:")
    print(f"  Decision Path : {path}")
    print(f"  Predicted     : {pred}")
    print(f"  Actual Class  : {t['Actual']}\n")

# ==========================================
# TASK 5: CONFUSION MATRIX & EVALUATION
# ==========================================
print("--- TASK 5: Confusion Matrix & Evaluation ---")
# Индексы классов для матрицы: 0=ON_TIME, 1=DELAYED, 2=SEVERE_DELAYED
class_to_idx = {cls: i for i, cls in enumerate(classes)}
matrix = [[0]*3 for _ in range(3)]

for _, actual, pred in predictions:
    row = class_to_idx[actual]
    col = class_to_idx[pred]
    matrix[row][col] += 1

# Вывод матрицы
print(f"{'Actual \\ Pred':<16} | {'ON_TIME':<10} | {'DELAYED':<10} | {'SEVERE_DELAYED':<15}")
print("-" * 60)
for i, cls_row in enumerate(classes):
    print(f"{cls_row:<16} | {matrix[i][0]:<10} | {matrix[i][1]:<10} | {matrix[i][2]:<15}")

# Расчет Accuracy
correct = sum(matrix[i][i] for i in range(3))
total_test = len(test_parcels)
accuracy = correct / total_test
print(f"\nModel Accuracy: {accuracy:.4f} ({correct}/{total_test})")

# Поиск наиболее часто путаемого класса
confusions = {}
for i in range(3):
    for j in range(3):
        if i != j and matrix[i][j] > 0:
            confusions[(classes[i], classes[j])] = matrix[i][j]

if confusions:
    most_confused = max(confusions, key=confusions.get)
    print(f"Most confused case: Actual '{most_confused[0]}' was misclassified as '{most_confused[1]}' ({confusions[most_confused]} times).")
else:
    print("Most confused case: None (Perfect classification on the test set).")
