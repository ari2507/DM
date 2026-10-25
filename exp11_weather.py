# QUESTION:
# Implement Mixed Naive Bayes using the Weather dataset.
# Use categorical probabilities with Laplace smoothing for Outlook and Windy,
# and Gaussian probability distributions for Temperature and Humidity.
# Predict the class for Outlook=sunny, Temperature=66, Humidity=90,
# and Windy=TRUE. Evaluate the model using 14-fold Leave-One-Out
# Cross-Validation (LOOCV), accuracy, precision, recall, F1-score,
# and a confusion matrix.

# CODE:

import pandas as pd
import math
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load the dataset
df = pd.read_csv("weather_numeric - weather_numeric.csv")
df.columns = df.columns.str.strip().str.lower()

# Clean categorical values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip().str.lower()

# Convert numerical columns
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

# Remove rows containing missing required values
required = ["outlook", "temperature", "humidity", "windy", "play"]
df = df.dropna(subset=required).reset_index(drop=True)

target = "play"
categorical_attributes = ["outlook", "windy"]
numeric_attributes = ["temperature", "humidity"]

print("WEATHER DATASET")
print(df.to_string(index=False))


# Calculate Gaussian probability
def gaussian_probability(x, mean, variance):
    variance = max(variance, 1e-9)

    return (
        1 / math.sqrt(2 * math.pi * variance)
    ) * math.exp(
        -((x - mean) ** 2) / (2 * variance)
    )


# Train the Mixed Naive Bayes model
def train_model(data):
    model = {}

    model["classes"] = sorted(data[target].unique())
    model["total"] = len(data)
    model["counts"] = data[target].value_counts().to_dict()
    model["cat_prob"] = {}
    model["num_stats"] = {}

    for cls in model["classes"]:
        class_data = data[data[target] == cls]
        class_count = len(class_data)

        model["cat_prob"][cls] = {}
        model["num_stats"][cls] = {}

        # Calculate categorical probabilities with Laplace smoothing
        for attr in categorical_attributes:
            values = sorted(data[attr].unique())
            counts = class_data[attr].value_counts()

            model["cat_prob"][cls][attr] = {}

            for value in values:
                probability = (
                    counts.get(value, 0) + 1
                ) / (class_count + len(values))

                model["cat_prob"][cls][attr][value] = probability

        # Calculate mean and population variance for numeric attributes
        for attr in numeric_attributes:
            mean = class_data[attr].mean()
            variance = class_data[attr].var(ddof=0)

            model["num_stats"][cls][attr] = (mean, variance)

    return model


# Predict one record
def predict_one(model, sample):
    scores = {}

    for cls in model["classes"]:
        # Start with the class prior probability
        score = model["counts"][cls] / model["total"]

        # Multiply categorical likelihoods
        for attr in categorical_attributes:
            value = str(sample[attr]).strip().lower()
            probabilities = model["cat_prob"][cls][attr]

            if value in probabilities:
                score *= probabilities[value]
            else:
                score *= 1 / (
                    model["counts"][cls] + len(probabilities) + 1
                )

        # Multiply Gaussian likelihoods
        for attr in numeric_attributes:
            mean, variance = model["num_stats"][cls][attr]

            score *= gaussian_probability(
                float(sample[attr]), mean, variance
            )

        scores[cls] = score

    predicted_class = max(scores, key=scores.get)
    total_score = sum(scores.values())

    # Normalize scores to obtain relative posterior probabilities
    if total_score > 0:
        posterior = {
            cls: score / total_score
            for cls, score in scores.items()
        }
    else:
        posterior = {cls: 0 for cls in scores}

    return predicted_class, scores, posterior


# Train using the complete dataset
model = train_model(df)

# Display class prior probabilities
print("\nCLASS PRIOR PROBABILITIES")

for cls in model["classes"]:
    prior = model["counts"][cls] / model["total"]
    print(cls, "=", round(prior, 4))


# Display categorical probability tables
print("\nCATEGORICAL PROBABILITY TABLES")

for cls in model["classes"]:
    print("\nClass:", cls)

    for attr in categorical_attributes:
        print("\nAttribute:", attr)

        for value, probability in model["cat_prob"][cls][attr].items():
            print(value, "=", round(probability, 4))


# Display numerical statistics
print("\nNUMERICAL STATISTICS")

for cls in model["classes"]:
    print("\nClass:", cls)

    for attr in numeric_attributes:
        mean, variance = model["num_stats"][cls][attr]

        print(
            attr,
            "Mean =", round(mean, 4),
            "Variance =", round(variance, 4)
        )


# Predict the new weather record
new_weather = {
    "outlook": "sunny",
    "temperature": 66,
    "humidity": 90,
    "windy": "true"
}

prediction, scores, posterior = predict_one(model, new_weather)

print("\nNEW WEATHER RECORD")
print(new_weather)

print("\nNAIVE BAYES SCORES")

for cls, score in scores.items():
    print(cls, "=", score)

print("\nNORMALIZED POSTERIOR PROBABILITIES")

for cls, probability in posterior.items():
    print(cls, "=", round(probability, 4))

print("\nPREDICTED CLASS:", prediction)


# --------------------------------------------------
# LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# --------------------------------------------------

actual_values = []
predicted_values = []

for i in range(len(df)):
    # Train on all rows except the current row
    train_data = df.drop(index=i).reset_index(drop=True)

    # Test on the held-out row
    test_sample = df.iloc[i]

    fold_model = train_model(train_data)

    fold_prediction, _, _ = predict_one(
        fold_model, test_sample
    )

    actual_values.append(test_sample[target])
    predicted_values.append(fold_prediction)


# Calculate evaluation metrics
accuracy = accuracy_score(actual_values, predicted_values)

precision = precision_score(
    actual_values,
    predicted_values,
    pos_label="yes",
    zero_division=0
)

recall = recall_score(
    actual_values,
    predicted_values,
    pos_label="yes",
    zero_division=0
)

f1 = f1_score(
    actual_values,
    predicted_values,
    pos_label="yes",
    zero_division=0
)

cm = confusion_matrix(
    actual_values,
    predicted_values,
    labels=["yes", "no"]
)

print("\nLOOCV RESULTS")
print("Number of folds:", len(df))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nCONFUSION MATRIX")

cm_df = pd.DataFrame(
    cm,
    index=["Actual yes", "Actual no"],
    columns=["Predicted yes", "Predicted no"]
)

print(cm_df.to_string())



# ============================================================
# EXPLANATION, FORMULAS AND CONCEPTS
# ============================================================

# 1. NAIVE BAYES
#
# Naive Bayes is a supervised classification algorithm
# based on Bayes' theorem.
#
# Formula:
#
# P(C | X) = P(C) * P(X | C) / P(X)
#
# C = target class, such as Play = Yes.
# X = observed input attributes.
#
# Since P(X) is the same for all classes, classification
# compares P(C) * P(X | C).


# 2. CONDITIONAL INDEPENDENCE ASSUMPTION
#
# Naive Bayes assumes that the input attributes are
# conditionally independent given the target class.
#
# Therefore:
#
# P(X | C) = P(X1 | C) * P(X2 | C) * ... * P(Xn | C)
#
# For this experiment:
#
# Score(C) = P(C)
#          * P(Outlook | C)
#          * P(Windy | C)
#          * P(Temperature | C)
#          * P(Humidity | C)


# 3. PRIOR PROBABILITY
#
# Prior probability is the probability of a class
# before considering the new input.
#
# Formula:
#
# P(C) = Number of records in class C / Total records


# 4. LAPLACE SMOOTHING
#
# Outlook and Windy are categorical attributes.
# Their probabilities are calculated from observed frequencies.
#
# Formula:
#
# P(value | C) = (count(value, C) + 1) / (count(C) + K)
#
# K = Number of possible categories for the attribute.
#
# Adding 1 prevents zero probabilities.
# This is called Laplace smoothing or add-one smoothing.


# 5. GAUSSIAN LIKELIHOOD
#
# Temperature and Humidity are numerical attributes.
# They are modeled using Gaussian distributions.
#
# Formula:
#
# P(x | C) =
#   1 / sqrt(2 * pi * variance)
#   * exp(-(x - mean)**2 / (2 * variance))
#
# Mean = Average value for the attribute within class C.
#
# Variance = Average squared difference from the mean.
#
# The program uses population variance, calculated with ddof=0.
#
# Note: This is a probability density, not the probability
# of one exact continuous numerical value.


# 6. CLASSIFICATION
#
# Calculate a score for every class.
#
# Score(C) = Prior * All four likelihoods
#
# Predict the class with the largest score.
#
# Normalized posterior probability:
#
# P(C | X) = Score(C) / Sum of all class scores
#
# The normalization makes the posterior probabilities
# sum to 1, provided the total score is positive.


# 7. LEAVE-ONE-OUT CROSS-VALIDATION
#
# If the dataset has 14 records, there are 14 folds.
#
# For each fold:
# - Train on 13 records.
# - Test on the remaining record.
# - Repeat until every record has been tested once.
#
# This evaluates the classifier using all records as test data
# without training on the particular record being predicted.


# 8. CONFUSION MATRIX
#
# Rows represent actual classes.
# Columns represent predicted classes.
#
# TP = Actual Yes and predicted Yes.
# TN = Actual No and predicted No.
# FP = Actual No but predicted Yes.
# FN = Actual Yes but predicted No.


# 9. EVALUATION FORMULAS
#
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
#
# Precision = TP / (TP + FP)
#
# Recall = TP / (TP + FN)
#
# F1-score = 2 * Precision * Recall / (Precision + Recall)
#
# Precision, recall, and F1-score are calculated for Play = Yes.