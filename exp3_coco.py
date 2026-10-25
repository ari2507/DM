# Part A - Normalisation (Exp 3), Cocoa: Apply Min-Max normalisation to [0, 1] and Z-score normalisation to Rating and Cocoa %. Plot histograms before and after, and explain why the shape of the distribution does not change
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")

# Convert Cocoa Percent to numbers
df["Cocoa Percent"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

cols = ["Rating", "Cocoa Percent"]

# Histograms before normalisation
df[cols].hist()
plt.suptitle("Before Normalisation")
plt.tight_layout()
plt.show()

# Min-Max Normalisation (0 to 1)
minmax = MinMaxScaler()
df[cols + []] = minmax.fit_transform(df[cols])

# Histograms after Min-Max normalisation
df[cols].hist()
plt.suptitle("After Min-Max Normalisation")
plt.tight_layout()
plt.show()

# Reload original data for Z-score normalisation
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")
df["Cocoa Percent"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# Z-score Normalisation
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

# Histograms after Z-score normalisation
df[cols].hist()
plt.suptitle("After Z-score Normalisation")
plt.tight_layout()
plt.show()


# . Aim of the experiment
# To apply Min-Max normalisation and Z-score normalisation to the Rating and Cocoa Percent columns of a cocoa dataset, plot histograms before and after normalisation, and understand how normalisation affects the distribution.
# 2. What is normalisation?
# Normalisation is a data preprocessing technique that transforms numerical values into a different scale.

# For example, a dataset might contain:
# - Rating: values between 1 and 5.
# - Cocoa Percent: values between 40 and 100.
# Since these columns have different ranges, some machine learning algorithms may be affected by their scales. Normalisation helps make the scales comparable.

# Why is it needed?
# 1. To bring numerical features onto comparable scales.
# 2. To prevent large-scale features from dominating distance-based algorithms.
# 3. To improve the usefulness of data preprocessing in machine learning pipelines.
# There are two techniques in this experiment:
# - Min-Max normalisation
# - Z-score normalisation (standardisation)
# 3. Min-Max normalisation
# Min-Max normalisation transforms values into a specified range, usually 0 to 1.
# Formula
# \[
# x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
# \]

# Where:
# - \(x\) = original value
# - \(x_{\min}\) = minimum value in the column
# - \(x_{\max}\) = maximum value in the column
# - \(x'\) = normalised value
# Example
# Suppose the ratings in a small dataset are:
# \[
# [2,\ 3,\ 4,\ 5]
# \]

# Minimum = 2 and maximum = 5.

# For the value \(x=4\):
# \[
# x'=\frac{4-2}{5-2}=\frac{2}{3}=0.667
# \]

# The complete transformation is:
# Original value	Min-Max value
# 2	0.000
# 3	0.333
# 4	0.667
# 5	1.000

# The minimum becomes 0, the maximum becomes 1, and the other values fall between them.
# Python code
# from sklearn.preprocessing import MinMaxScalerscaler = MinMaxScaler()df[cols] = scaler.fit_transform(df[cols])



# Explanation:
# - MinMaxScaler() creates the normalisation object.
# - fit_transform() finds the minimum and maximum of each selected column and transforms the values.
# - df[cols] stores the transformed values back into the DataFrame.
# The transformation is performed separately for Rating and Cocoa Percent, so each column gets its own minimum and maximum.
# 4. Z-score normalisation
# Z-score normalisation, also called standardisation, transforms values so that each feature has a mean of approximately 0 and a standard deviation of approximately 1.
# Formula
# \[
# z=\frac{x-\mu}{\sigma}
# \]

# Where:
# - \(x\) = original value
# - \(\mu\) = mean of the column
# - \(\sigma\) = standard deviation of the column
# - \(z\) = standardised value
# Example
# Suppose a column has:
# - Mean = 3
# - Standard deviation = 1
# For an observation \(x=5\):
# \[
# z=\frac{5-3}{1}=2
# \]

# This means the observation is two standard deviations above the mean.

# For \(x=2\):
# \[
# z=\frac{2-3}{1}=-1
# \]

# This means the observation is one standard deviation below the mean.
# How to interpret Z-scores
# Z-score	Meaning
# 0	Equal to the mean
# +1	One standard deviation above the mean
# -1	One standard deviation below the mean
# +2	Two standard deviations above the mean
# -2	Two standard deviations below the mean

# A negative Z-score does not mean the original value was negative. It means the value is below the mean.
# Python code
# from sklearn.preprocessing import StandardScalerscaler = StandardScaler()df[cols] = scaler.fit_transform(df[cols])



# Explanation:
# - StandardScaler() creates the standardisation object.
# - fit_transform() calculates the mean and standard deviation for each column and applies the Z-score formula.
# - The transformed values replace the original values in the selected columns.
# By default, StandardScaler uses the population-style standard deviation, equivalent to ddof=0 in pandas.
# 5. What is a histogram?
# A histogram is a graph that shows the frequency distribution of numerical data.
# - X-axis: Values or intervals (bins).
# - Y-axis: Number of observations in each interval.
# - Bars: Show how many observations fall within each interval.
# For example, a cocoa rating histogram helps you see whether most ratings are concentrated around 3, 3.5, or 4.
# Code
# df[cols].hist()plt.suptitle("Before Normalisation")plt.tight_layout()plt.show()



# Explanation:
# - df[cols] selects the two numerical columns.
# - .hist() creates a histogram for each selected column.
# - plt.suptitle() adds a common title.
# - plt.tight_layout() adjusts spacing.
# - plt.show() displays the plots.
# The same code is used after normalisation to compare the results.
# 6. Why does the shape of the distribution not change?
# Both Min-Max and Z-score normalisation use a positive linear transformation of the form
# \[
# x'=ax+b,\qquad a>0
# \]

# This changes the horizontal scale but preserves the ordering of observations and their relative spacing proportionally.

# For Min-Max normalisation:
# \[
# x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
# \]

# For Z-score normalisation:
# \[
# z=\frac{x-\mu}{\sigma}
# \]

# The values change, but the transformation does not make a skewed distribution symmetric or turn a non-normal distribution into a normal distribution.

# For example, if most cocoa ratings are concentrated at the higher end before normalisation, they will remain concentrated toward the corresponding higher end afterward.

# Important: The plotted histograms may not look exactly identical because the axes, bin boundaries, and automatic bin selection can change. Normalisation does not guarantee that every histogram bar has the same height.
# 7. Complete concept comparison
# Property	Min-Max normalisation	Z-score normalisation
# Formula	\(\frac{x-x_{\min}}{x_{\max}-x_{\min}}\)	\(\frac{x-\mu}{\sigma}\)
# Typical range	0 to 1	No fixed range
# Mean	Not necessarily 0	Approximately 0
# Standard deviation	Not necessarily 1	Approximately 1
# Uses	Minimum and maximum	Mean and standard deviation
# Effect of outliers	Can be strongly affected	Can also be affected
# Changes distribution shape?	No, not mathematically	No, not mathematically

# A practical note: for model evaluation, fit the scaler on training data and apply that same fitted scaler to test data. This avoids data leakage.