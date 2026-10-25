```python
import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")

# 2. Convert Cocoa Percent to numeric values
df["Cocoa Percent"] = (
    df["Cocoa Percent"].astype(str).str.replace("%", "", regex=False)
)
df["Cocoa Percent"] = pd.to_numeric(df["Cocoa Percent"], errors="coerce")

# Remove missing values and sort the values
data = df[["Cocoa Percent"]].dropna().sort_values(
    "Cocoa Percent"
).reset_index(drop=True)

# 3. Create 4 equal-width bins
minimum = data["Cocoa Percent"].min()
maximum = data["Cocoa Percent"].max()

edges = np.linspace(minimum, maximum, 5)
data["Bin"] = pd.cut(
    data["Cocoa Percent"],
    bins=edges,
    labels=False,
    include_lowest=True
)

# 4. Display bin ranges and counts
print("Bin ranges and counts:")
for i in range(4):
    lower = edges[i]
    upper = edges[i + 1]
    count = (data["Bin"] == i).sum()
    print(f"Bin {i + 1}: {lower:.2f} to {upper:.2f}, Count = {count}")

# 5. Calculate bin means and medians
data["Bin Mean"] = data.groupby("Bin")["Cocoa Percent"].transform("mean")
data["Bin Median"] = data.groupby("Bin")["Cocoa Percent"].transform("median")

# 6. Smooth values using bin boundaries
def smooth_boundary(row):
    bin_number = int(row["Bin"])
    lower = edges[bin_number]
    upper = edges[bin_number + 1]
    value = row["Cocoa Percent"]

    if abs(value - lower) <= abs(value - upper):
        return lower
    else:
        return upper

data["Bin Boundary"] = data.apply(smooth_boundary, axis=1)

# 7. Display first 15 sorted records with all smoothing methods
print("\nFirst 15 records after sorting and smoothing:")
print(
    data[[
        "Cocoa Percent",
        "Bin",
        "Bin Mean",
        "Bin Median",
        "Bin Boundary"
    ]].head(15).round(2).to_string(index=False)
)
```


# Explanation of the code
# 1. What is equal-width binning?
# Equal-width binning is a data preprocessing technique that divides the range of numerical values into a fixed number of intervals called bins.

# In this experiment, the Cocoa Percent values are divided into four bins of equal width.

# The formula is:
# \[
# \text{Bin width}=\frac{\text{Maximum}-\text{Minimum}}{\text{Number of bins}}
# \]

# Example: Suppose the minimum cocoa percentage is 50 and the maximum is 100.
# \[
# \text{Bin width}=\frac{100-50}{4}=12.5
# \]

# The four bin ranges are:
# Bin	Range
# 1	50–62.5
# 2	62.5–75
# 3	75–87.5
# 4	87.5–100

# These are example values; your actual bin ranges will be calculated from your dataset.
# 2. Explanation of each code section
# Step 1 — Load the dataset
# df = pd.read_csv("flavors_of_cocoa (1) (2).csv")



# Reads the Cocoa CSV file into a DataFrame.

# Step 2 — Convert cocoa percentage to numbers
# df["Cocoa Percent"] = (    df["Cocoa Percent"].astype(str).str.replace("%", "", regex=False))df["Cocoa Percent"] = pd.to_numeric(df["Cocoa Percent"], errors="coerce")



# Removes the % symbol and converts the column to numerical values. Invalid entries become missing values.

# Step 3 — Sort the data
# data = df[["Cocoa Percent"]].dropna().sort_values(    "Cocoa Percent").reset_index(drop=True)



# - Selects the required column.
# - Removes missing values.
# - Sorts the percentages in ascending order.
# - Resets the row index.
# Step 4 — Create four equal-width bins
# edges = np.linspace(minimum, maximum, 5)



# np.linspace() creates five boundary points, which define four intervals. pd.cut() assigns each value to its appropriate bin.

# Step 5 — Display bin ranges and counts

# The loop prints the lower and upper boundary of each bin and counts how many observations fall into it.

# Note that equal-width bins have equal interval widths, but their counts do not have to be equal.
# 3. Smoothing techniques and formulas
# Smoothing reduces small fluctuations in numerical data by replacing values with representative values.

# A. Smoothing by bin means

# Each value is replaced by the mean of its bin.
# \[
# \text{Mean}=\frac{x_1+x_2+\cdots+x_n}{n}
# \]

# Example: for values \(60,62,64\),
# \[
# \text{Mean}=\frac{60+62+64}{3}=62
# \]

# All three values become 62.

# B. Smoothing by bin medians

# Each value is replaced by the median of its bin.

# The median is the middle value after sorting. For an even number of values, it is the average of the two middle values.

# Example: for \(60,62,70\), the median is 62, so all three values become 62.

# C. Smoothing by bin boundaries

# Each value is replaced by the closest boundary of its bin.

# For a value \(x\), with lower boundary \(L\) and upper boundary \(U\):
# \[
# x_{\text{smooth}}=
# \begin{cases}
# L,& |x-L|\leq|x-U|\\
# U,& |x-L|>|x-U|
# \end{cases}
# \]

# If a value is equally distant from both boundaries, this code chooses the lower boundary.

# The code stores the three results in Bin Mean, Bin Median, and Bin Boundary.
# 4. Understanding the output
# The final table displays the first 15 sorted records with these columns:
# Column	Meaning
# Cocoa Percent	Original cocoa percentage
# Bin	Assigned bin, numbered 0–3
# Bin Mean	Value after mean smoothing
# Bin Median	Value after median smoothing
# Bin Boundary	Value after boundary smoothing

# The first 15 records are shown for convenient comparison; the smoothing operations are applied to the entire cleaned dataset.
# Viva questions and answers
# 1. What is binning?

# Binning divides continuous numerical values into intervals or groups.

# 2. What is equal-width binning?

# It divides the total numerical range into bins with the same width.

# 3. What is the formula for bin width?
# \[
# \frac{\text{Maximum}-\text{Minimum}}{\text{Number of bins}}
# \]

# 4. What is smoothing?

# Smoothing replaces original values with representative values to reduce fluctuations.

# 5. What is smoothing by bin means?

# Every value is replaced by the average of the values in its bin.

# 6. What is smoothing by bin medians?

# Every value is replaced by the median of its bin.

# 7. What is smoothing by bin boundaries?

# Every value is replaced by whichever boundary of its bin is closest.

# 8. Do equal-width bins contain equal numbers of observations?

# No. Their widths are equal, but their counts can differ.

# 9. What is the difference between equal-width and equal-frequency binning?

# Equal-width binning uses intervals of equal numerical width. Equal-frequency binning aims to put approximately equal numbers of observations in each bin.

# 10. Why is binning used?

# It can simplify data, reduce the effect of small variations, and make numerical patterns easier to analyze.
# Conclusion
# The Cocoa Percent values were cleaned, sorted, and divided into four equal-width bins. The range and count of each bin were displayed. The original values were then smoothed using bin means, bin medians, and bin boundaries, and the first 15 records were presented for comparison.