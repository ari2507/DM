
# Question:
# Apply Min-Max normalisation [0,1] and Z-score normalisation
# to Fam and Health attributes in the GHI dataset.
# Display the first 10 original and normalised values and
# verify the statistics after normalisation.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Load the dataset
df = pd.read_csv("GHI_Report (1).csv")
df.columns = df.columns.str.strip()

# 2. Select the required columns
columns = ["Fam", "Health"]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values in Fam or Health
data = df[columns].dropna().reset_index(drop=True)

print("Number of records:", len(data))

# 3. Apply Min-Max normalisation to [0, 1]
minmax_scaler = MinMaxScaler()
minmax_values = minmax_scaler.fit_transform(data)

minmax_df = pd.DataFrame(
    minmax_values,
    columns=["Fam_MinMax", "Health_MinMax"]
)

# 4. Apply Z-score normalisation
zscore_scaler = StandardScaler()
zscore_values = zscore_scaler.fit_transform(data)

zscore_df = pd.DataFrame(
    zscore_values,
    columns=["Fam_Zscore", "Health_Zscore"]
)

# 5. Display first 10 original and normalised values
result = pd.concat(
    [
        data.rename(columns={
            "Fam": "Fam_Original",
            "Health": "Health_Original"
        }),
        minmax_df,
        zscore_df
    ],
    axis=1
)

print("\nFirst 10 original and normalised values:")
print(result.head(10).round(4).to_string(index=False))

# 6. Verify statistics after normalisation
print("\n===== STATISTICS AFTER NORMALISATION =====")

for name, values in [
    ("Original", data),
    ("Min-Max [0,1]", minmax_df),
    ("Z-score", zscore_df)
]:
    print(f"\n{name}:")
    print("Mean:")
    print(values.mean().round(4))
    print("Standard deviation (population):")
    print(values.std(ddof=0).round(4))
    print("Minimum:")
    print(values.min().round(4))
    print("Maximum:")
    print(values.max().round(4))

# Explanation:
# Min-Max formula:
# X_new = (X - X_min) / (X_max - X_min)
# It transforms values into the range [0, 1].
#
# Z-score formula:
# Z = (X - Mean) / Standard Deviation
# It transforms values to have mean approximately 0
# and population standard deviation approximately 1.
#
# Min-Max is useful when a fixed range is needed.
# Z-score is useful when scaling values relative to their
# mean and standard deviation, such as before K-Means clustering.
