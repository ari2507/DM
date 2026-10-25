
# QUESTION:
# Implement Decision Tree classification using the Breast Cancer Wisconsin
# (Diagnostic) dataset to predict Diagnosis (M = Malignant, B = Benign)
# from 30 numerical features.
#
# 1. Split the dataset into 80% training and 20% testing.
# 2. Train two Decision Trees with max_depth=4:
#    a) criterion="gini"
#    b) criterion="entropy"
# 3. Plot both Decision Trees.
# 4. Compare their root attributes and top 5 feature importances.
# 5. Evaluate both models using accuracy, precision, recall, F1-score,
#    and confusion matrix.
# 6. State which model is preferred and explain why recall for
#    malignant cases is especially important.

# CODE:

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: LOAD THE DATASET
# --------------------------------------------------

df = pd.read_csv("breast_cancer_wisconsin - breast_cancer_wisconsin.csv")

print("FIRST 5 ROWS OF THE DATASET")
print(df.head().to_string(index=False))

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Remove rows containing missing values, if any
df = df.dropna().reset_index(drop=True)

# Diagnosis is the target column
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].astype(str).str.strip().str.upper()

# Display class counts
print("\nDIAGNOSIS CLASS COUNTS")
print(y.value_counts().to_string())

print("\nNumber of features:", X.shape[1])


# --------------------------------------------------
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().to_string())

print("\nTesting class distribution:")
print(y_test.value_counts().to_string())


# --------------------------------------------------
# STEP 3: TRAIN TWO DECISION TREE MODELS
# --------------------------------------------------

# Model 1: Gini criterion
tree_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

tree_gini.fit(X_train, y_train)


# Model 2: Entropy criterion
tree_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

tree_entropy.fit(X_train, y_train)


# --------------------------------------------------
# STEP 4: PLOT BOTH DECISION TREES
# --------------------------------------------------

plt.figure(figsize=(22, 10))

plot_tree(
    tree_gini,
    feature_names=X.columns,
    class_names=["B", "M"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Using Gini Criterion")
plt.tight_layout()
plt.show()


plt.figure(figsize=(22, 10))

plot_tree(
    tree_entropy,
    feature_names=X.columns,
    class_names=["B", "M"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Using Entropy Criterion")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# STEP 5: COMPARE ROOT ATTRIBUTES
# --------------------------------------------------

gini_root_index = tree_gini.tree_.feature[0]
entropy_root_index = tree_entropy.tree_.feature[0]

if gini_root_index >= 0:
    gini_root = X.columns[gini_root_index]
else:
    gini_root = "No split"

if entropy_root_index >= 0:
    entropy_root = X.columns[entropy_root_index]
else:
    entropy_root = "No split"

print("\nROOT ATTRIBUTE COMPARISON")
print("Gini root attribute:", gini_root)
print("Entropy root attribute:", entropy_root)

if gini_root == entropy_root:
    print("Both trees have the same root attribute.")
else:
    print("The trees have different root attributes.")


# --------------------------------------------------
# STEP 6: TOP 5 FEATURE IMPORTANCES
# --------------------------------------------------

gini_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_gini.feature_importances_
}).sort_values(by="Importance", ascending=False)

entropy_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_entropy.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTOP 5 FEATURES - GINI TREE")
print(gini_importance.head(5).to_string(
    index=False,
    formatters={"Importance": "{:.4f}".format}
))

print("\nTOP 5 FEATURES - ENTROPY TREE")
print(entropy_importance.head(5).to_string(
    index=False,
    formatters={"Importance": "{:.4f}".format}
))


# --------------------------------------------------
# STEP 7: EVALUATE BOTH MODELS
# --------------------------------------------------

def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)

    # Malignant (M) is the positive class
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        pos_label="M",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        pos_label="M",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        pos_label="M",
        zero_division=0
    )

    print("\n====================================")
    print(model_name)
    print("====================================")

    print("Accuracy:", round(accuracy, 4))
    print("Precision for Malignant:", round(precision, 4))
    print("Recall for Malignant:", round(recall, 4))
    print("F1-score for Malignant:", round(f1, 4))

    # Rows = actual classes; columns = predicted classes
    # Explicit order: Malignant, Benign
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["M", "B"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )

    print("\nCONFUSION MATRIX")
    print(cm_df.to_string())

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        y_pred,
        labels=["M", "B"],
        target_names=["Malignant (M)", "Benign (B)"],
        zero_division=0
    ))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Malignant Precision": precision,
        "Malignant Recall": recall,
        "Malignant F1-score": f1
    }


gini_results = evaluate_model(
    tree_gini,
    "GINI DECISION TREE"
)

entropy_results = evaluate_model(
    tree_entropy,
    "ENTROPY DECISION TREE"
)


# --------------------------------------------------
# STEP 8: COMPARE EVALUATION MEASURES
# --------------------------------------------------

comparison = pd.DataFrame([
    gini_results,
    entropy_results
])

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison.round(4).to_string(index=False))


# --------------------------------------------------
# STEP 9: SELECT THE PREFERRED MODEL
# --------------------------------------------------

# In this medical classification task, malignant recall is the priority.
if gini_results["Malignant Recall"] > entropy_results["Malignant Recall"]:
    preferred = "Gini"
elif entropy_results["Malignant Recall"] > gini_results["Malignant Recall"]:
    preferred = "Entropy"
else:
    # If recall ties, compare malignant F1-score
    if gini_results["Malignant F1-score"] >= entropy_results["Malignant F1-score"]:
        preferred = "Gini"
    else:
        preferred = "Entropy"

print("\nPREFERRED MODEL BASED ON MALIGNANT RECALL")
print(preferred, "Decision Tree")

print("\nReason:")
print(
    "Recall for malignant cases is prioritised because missing a malignant "
    "tumour can delay diagnosis and treatment. If malignant recall is tied, "
    "the malignant F1-score is used as a secondary comparison."
)


# --------------------------------------------------
# EXPLANATION:
#
# 1. DATASET:
#    The dataset contains 30 numerical features describing cell nuclei
#    and a diagnosis column. M means Malignant and B means Benign.
#
# 2. TRAIN-TEST SPLIT:
#    80% of the data is used for training and 20% for testing.
#    random_state=42 makes the split reproducible.
#    stratify=y preserves the approximate class proportions in both sets.
#
# 3. GINI CRITERION:
#    Gini impurity measures how mixed the classes are at a node.
#    Gini = 1 - sum(p_i^2)
#    A split is preferred when it reduces the weighted Gini impurity.
#
# 4. ENTROPY CRITERION:
#    Entropy measures the uncertainty at a node.
#    Entropy = -sum(p_i * log2(p_i))
#    Information Gain = Parent Entropy - Weighted Child Entropy.
#
# 5. MAXIMUM DEPTH:
#    max_depth=4 restricts the tree to a maximum depth of four.
#    This keeps the trees relatively simple and helps limit overfitting.
#
# 6. TREE PLOTS:
#    plot_tree() displays the decision rules.
#    Each internal node shows the splitting condition, impurity,
#    sample count, and class distribution.
#
# 7. ROOT ATTRIBUTE:
#    The root is the first feature used to split the dataset.
#    Gini and Entropy may choose different root attributes because
#    they measure impurity differently.
#
# 8. FEATURE IMPORTANCE:
#    Feature importance estimates how much each feature contributes
#    to reducing impurity throughout the fitted tree.
#    Larger values indicate greater importance within that model.
#    Importances are model-specific and do not prove causation.
#
# 9. ACCURACY:
#    Accuracy = (TP + TN) / (TP + TN + FP + FN)
#    It measures the proportion of all predictions that are correct.
#
# 10. PRECISION FOR MALIGNANT:
#     Precision = TP / (TP + FP)
#     It measures how many cases predicted as malignant are actually
#     malignant.
#
# 11. RECALL FOR MALIGNANT:
#     Recall = TP / (TP + FN)
#     It measures how many actual malignant cases are correctly detected.
#
# 12. F1-SCORE FOR MALIGNANT:
#     F1 = 2 * Precision * Recall / (Precision + Recall)
#     It balances precision and recall.
#
# 13. CONFUSION MATRIX:
#     Rows show actual classes and columns show predicted classes.
#     The matrix is ordered M first and B second.
#     The bottom-left cell counts benign cases incorrectly predicted
#     as malignant; the top-right cell counts malignant cases missed
#     by the model.
#
# 14. WHY MALIGNANT RECALL MATTERS:
#     A false negative occurs when an actual malignant tumour is
#     predicted as benign. This may delay further investigation or
#     treatment. High malignant recall reduces these missed cases.
#
# 15. MODEL PREFERENCE:
#     The code prefers the model with higher malignant recall.
#     If recall ties, it compares malignant F1-score.
#     In real clinical use, the model must also be externally validated
#     and assessed by medical professionals; this experiment alone
#     is not sufficient for diagnosis.
#
# 16. PRINT:
#     print() is used for textual results instead of display().
#     Matplotlib is used to plot both trees.
# --------------------------------------------------
