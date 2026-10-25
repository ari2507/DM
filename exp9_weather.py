# QUESTION:
# Implement attribute selection using Information Gain and Gini Index
# for categorical and numerical attributes in the Weather dataset.
# For numerical attributes, test all possible midpoints.
# Display the results and identify the best splitting attribute.

# CODE:

import pandas as pd
import math

# Load the dataset
df = pd.read_csv("weather_numeric - weather_numeric.csv")
df.columns = df.columns.str.strip().str.lower()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

target = "play"

print("WEATHER DATASET")
print(df.to_string(index=False))


# Calculate entropy
def entropy(series):
    probabilities = series.value_counts(normalize=True)
    result = 0

    for p in probabilities:
        result -= p * math.log2(p)

    return result


# Calculate Gini Index
def gini(series):
    probabilities = series.value_counts(normalize=True)
    result = 1

    for p in probabilities:
        result -= p ** 2

    return result


# Calculate weighted entropy and Gini after splitting
def split_impurity(data, attribute, threshold=None):
    groups = []

    if threshold is not None:
        groups.append(data[data[attribute] <= threshold])
        groups.append(data[data[attribute] > threshold])
    else:
        for value in data[attribute].unique():
            groups.append(data[data[attribute] == value])

    weighted_entropy = 0
    weighted_gini = 0

    for group in groups:
        if len(group) > 0:
            weight = len(group) / len(data)
            weighted_entropy += weight * entropy(group[target])
            weighted_gini += weight * gini(group[target])

    return weighted_entropy, weighted_gini


parent_entropy = entropy(df[target])
parent_gini = gini(df[target])

print("\nParent Entropy:", round(parent_entropy, 4))
print("Parent Gini Index:", round(parent_gini, 4))

results = []

for attribute in df.columns:
    if attribute == target:
        continue

    if pd.api.types.is_numeric_dtype(df[attribute]):
        values = sorted(df[attribute].dropna().unique())

        # Test every midpoint between consecutive unique values
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            child_entropy, child_gini = split_impurity(
                df, attribute, threshold
            )

            information_gain = parent_entropy - child_entropy
            gini_reduction = parent_gini - child_gini

            results.append([
                attribute,
                "<= " + str(round(threshold, 3)),
                information_gain,
                child_gini,
                gini_reduction
            ])

    else:
        child_entropy, child_gini = split_impurity(df, attribute)

        information_gain = parent_entropy - child_entropy
        gini_reduction = parent_gini - child_gini

        results.append([
            attribute,
            "Categorical",
            information_gain,
            child_gini,
            gini_reduction
        ])


result_df = pd.DataFrame(
    results,
    columns=[
        "Attribute",
        "Split",
        "Information Gain",
        "Weighted Gini",
        "Gini Reduction"
    ]
)

print("\nALL ATTRIBUTE SPLIT RESULTS")
print(result_df.round(4).to_string(index=False))

best_ig = result_df.loc[result_df["Information Gain"].idxmax()]
best_gini = result_df.loc[result_df["Gini Reduction"].idxmax()]

print("\nBEST SPLIT USING INFORMATION GAIN")
print(best_ig.to_string())

print("\nBEST SPLIT USING GINI REDUCTION")
print(best_gini.to_string())

if best_ig["Attribute"] == best_gini["Attribute"]:
    print("\nBoth measures select the same attribute.")
else:
    print("\nThe measures select different attributes.")




# ============================================================
# EXPLANATION, FORMULAS AND CONCEPTS
# ============================================================

# 1. ENTROPY
#
# Entropy measures the impurity or uncertainty in the dataset.
#
# Formula:
#
# H(S) = -sum(p_i * log2(p_i))
#
# p_i = proportion of records belonging to class i.
#
# For the Play target:
#
# H(S) = -p(Yes)*log2(p(Yes)) - p(No)*log2(p(No))
#
# Entropy = 0 when all records belong to one class.
# Higher entropy means greater uncertainty.


# 2. INFORMATION GAIN
#
# Information Gain measures how much entropy decreases
# after splitting a dataset using an attribute.
#
# Formula:
#
# IG(S, A) = H(S) - H(S | A)
#
# Weighted entropy:
#
# H(S | A) = sum((size of group / total size) * H(group))
#
# The split with maximum Information Gain is preferred.


# 3. GINI INDEX
#
# Gini measures the impurity of a dataset.
#
# Formula:
#
# Gini(S) = 1 - sum(p_i ** 2)
#
# Weighted Gini:
#
# Gini_split = sum((size of group / total size) * Gini(group))
#
# The split with minimum weighted Gini is preferred.


# 4. CATEGORICAL ATTRIBUTES
#
# Outlook and Windy are categorical attributes.
#
# The dataset is divided into groups based on their values.
# For example, Outlook may have Sunny, Overcast, and Rainy groups.


# 5. NUMERICAL ATTRIBUTES
#
# Temperature and Humidity are numerical attributes.
#
# First, sort the distinct values.
#
# For every consecutive pair:
#
# Threshold = (value1 + value2) / 2
#
# Divide the dataset into:
#
# Group 1: value <= threshold
# Group 2: value > threshold
#
# Calculate Information Gain and weighted Gini for each split.


# 6. FINAL DECISION
#
# Maximum Information Gain gives the preferred split
# according to Information Gain.
#
# Minimum weighted Gini gives the preferred split
# according to Gini.
#
# Both methods may select the same split, but they
# are not guaranteed to agree.
#
# IMPORTANT:
# The code calculates the complete candidate-split table.
# The best split is selected from that table.