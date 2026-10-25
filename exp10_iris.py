# QUESTION:
# Implement a Decision Tree classifier on the Iris dataset.
# Split the data into 70% training and 30% testing using random_state=42.
# Display the tree, identify the root attribute, and evaluate the model
# using accuracy, precision, recall, F1-score, and a confusion matrix.

# CODE:

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("IRIS DATASET")
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = [iris.target_names[i] for i in y]
print(df.head(10).to_string(index=False))

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create and train the Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Display the tree in text format
print("\nDECISION TREE")
print(export_text(model, feature_names=list(iris.feature_names)))

# Identify the root attribute
root_index = model.tree_.feature[0]

if root_index >= 0:
    print("Root splitting attribute:", iris.feature_names[root_index])
else:
    print("The tree contains only a leaf node.")

# Predict the testing data
y_pred = model.predict(X_test)

print("\nACTUAL CLASSES")
print([iris.target_names[i] for i in y_test])

print("\nPREDICTED CLASSES")
print([iris.target_names[i] for i in y_pred])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("Accuracy percentage:", round(accuracy * 100, 2), "%")

# Calculate precision, recall, and F1-score
print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=list(iris.target_names),
        zero_division=0
    )
)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

cm_df = pd.DataFrame(
    cm,
    index=["Actual " + name for name in iris.target_names],
    columns=["Predicted " + name for name in iris.target_names]
)

print("\nCONFUSION MATRIX")
print(cm_df.to_string())




# ============================================================
# EXPLANATION, FORMULAS AND CONCEPTS
# ============================================================

# 1. DECISION TREE
#
# A Decision Tree is a supervised machine-learning algorithm.
# It predicts a target class using a sequence of conditions.
#
# Internal nodes contain conditions on input attributes.
# Branches represent the outcomes of those conditions.
# Leaf nodes contain the predicted class.


# 2. IRIS DATASET
#
# The four input attributes are:
#
# 1. Sepal length
# 2. Sepal width
# 3. Petal length
# 4. Petal width
#
# The target has three classes:
#
# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica


# 3. TRAIN-TEST SPLIT
#
# 70% of the records train the model.
# 30% of the records evaluate the model.
#
# random_state=42 makes the split reproducible.
#
# max_depth=3 restricts the maximum depth of the tree
# and helps control model complexity.


# 4. GINI IMPURITY
#
# The default splitting criterion is Gini impurity.
#
# Formula:
#
# Gini(S) = 1 - sum(p_i ** 2)
#
# The algorithm considers splits that reduce impurity.
# It selects a split based on the weighted impurity reduction.
#
# The tree's root attribute is the feature selected
# for the first split.


# 5. ACCURACY
#
# Accuracy = Correct predictions / Total predictions
#
# For binary classification, this is also:
#
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
#
# In multiclass classification, count all correct predictions
# and divide by the total number of predictions.


# 6. PRECISION
#
# Precision = TP / (TP + FP)
#
# It measures how many predictions for a class
# actually belong to that class.
#
# Precision is calculated separately for each Iris species.


# 7. RECALL
#
# Recall = TP / (TP + FN)
#
# It measures how many actual records of a class
# were correctly identified.


# 8. F1-SCORE
#
# F1 = 2 * Precision * Recall / (Precision + Recall)
#
# F1 combines precision and recall.
# A higher F1-score indicates a better balance between them.
#
# If both precision and recall are zero, F1 is zero.


# 9. CONFUSION MATRIX
#
# Rows represent actual classes.
# Columns represent predicted classes.
#
# The diagonal contains correct predictions.
# Off-diagonal entries contain incorrect predictions.
#
# The matrix is 3x3 because the dataset has three classes.


# 10. ROOT ATTRIBUTE
#
# The root attribute is the first feature used for splitting.
# The program extracts it directly from the trained tree.
#
# It is selected because its split gives a good reduction
# in impurity among the candidate splits considered by the tree.
#
# The exact root attribute is printed by the program.