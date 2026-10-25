
# Question:
# Perform box plot analysis on the GHI dataset.
# Calculate the five-number summary for Economy, Fam, Health,
# and Freedom using a user-defined function.
# Draw side-by-side box plots, detect outliers using the IQR method,
# and compare the spread of the attributes.

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the GHI dataset
df = pd.read_csv("GHI_Report (1).csv")
df.columns = df.columns.str.strip()

# 2. Select the required attributes
columns = ["Economy", "Fam", "Health", "Freedom"]

# Convert values to numeric
for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values in the selected attributes
data = df[columns].dropna()

print("Number of complete records:", len(data))

# 3. User-defined function for five-number summary
def five_number_summary(series):
    return pd.Series({
        "Minimum": series.min(),
        "Q1": series.quantile(0.25),
        "Median": series.median(),
        "Q3": series.quantile(0.75),
        "Maximum": series.max()
    })

print("\n===== FIVE-NUMBER SUMMARY =====")
summary = data.apply(five_number_summary).T
print(summary.round(3))

# 4. Calculate IQR and detect outliers
print("\n===== IQR AND OUTLIER ANALYSIS =====")

outlier_counts = {}

for col in columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[col] < lower_limit) |
        (data[col] > upper_limit)
    ][col]

    outlier_counts[col] = len(outliers)

    print(f"\nAttribute: {col}")
    print("Q1:", round(Q1, 3))
    print("Q3:", round(Q3, 3))
    print("IQR:", round(IQR, 3))
    print("Lower limit:", round(lower_limit, 3))
    print("Upper limit:", round(upper_limit, 3))
    print("Number of outliers:", len(outliers))
    print("Outlier values:", outliers.tolist())

# 5. Draw side-by-side box plots
plt.figure(figsize=(9, 6))

plt.boxplot(
    [data[col] for col in columns],
    tick_labels=columns,
    showmeans=True
)

plt.title("Box Plot Analysis of GHI Attributes")
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Compare the spread of attributes using IQR
print("\n===== SPREAD COMPARISON =====")

iqr_values = {
    col: data[col].quantile(0.75) - data[col].quantile(0.25)
    for col in columns
}

for col, iqr in sorted(
    iqr_values.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{col}: IQR = {iqr:.3f}")

widest_attribute = max(iqr_values, key=iqr_values.get)
narrowest_attribute = min(iqr_values, key=iqr_values.get)

print("\nLargest IQR:", widest_attribute)
print("Smallest IQR:", narrowest_attribute)

# Explanation:
# Five-number summary:
# Minimum, Q1, Median, Q3, Maximum.
#
# IQR = Q3 - Q1
#
# Lower outlier limit = Q1 - 1.5 * IQR
# Upper outlier limit = Q3 + 1.5 * IQR
#
# Values outside these limits are identified as outliers.
# A larger IQR indicates greater spread in the middle 50% of values.
# Box plots help compare medians, spread, skewness, and potential outliers.
