# Part B - Centroid and distances (Exp 12), GHI: Take data rows 1-6 (the first 6 records) with the attributes Economy, Fam, Health and Freedom. Compute the centroid and each point's Euclidean, Manhattan and Minkowski (p = 3) distance from it. Display the pairwise Euclidean distances as a lower triangular matrix, and name the nearest and farthest point under each measure.
import pandas as pd
import numpy as np

df = pd.read_csv("GHI_Report (1).csv")

cols = ["Economy", "Fam", "Health", "Freedom"]
data = df.iloc[:6][cols].astype(float)

print(data)
centroid = data.mean()

print("Centroid:")
print(centroid)
diff = abs(data - centroid)

data["Euclidean"] = np.sqrt((diff ** 2).sum(axis=1))
data["Manhattan"] = diff.sum(axis=1)
data["Minkowski p=3"] = (diff ** 3).sum(axis=1) ** (1/3)

print(data)
from scipy.spatial.distance import cdist

X = df.iloc[:6][cols].astype(float).values

D = cdist(X, X, metric="euclidean")

# Keep only the lower triangular matrix
lower = np.tril(D)

print(pd.DataFrame(lower, index=data.index, columns=data.index))
dist_cols = ["Euclidean", "Manhattan", "Minkowski p=3"]

for col in dist_cols:
    nearest = data[col].idxmin()
    farthest = data[col].idxmax()

    print("\n", col)
    print("Nearest point:", nearest, "Distance:", data.loc[nearest, col])
    print("Farthest point:", farthest, "Distance:", data.loc[farthest, col])


# . Aim of the experiment
# To take the first six records from the GHI dataset, use the attributes Economy, Fam, Health, and Freedom to calculate the centroid, find each record's Euclidean, Manhattan, and Minkowski distance from that centroid, construct a lower triangular pairwise Euclidean distance matrix, and identify the nearest and farthest records from the centroid under each measure.
# 2. What is a centroid?
# A centroid is the point representing the mean position of a group of observations.

# For this experiment, each observation has four features:
# \[
# X_i=(E_i,F_i,H_i,R_i)
# \]

# Where:
# - \(E_i\) = Economy.
# - \(F_i\) = Fam.
# - \(H_i\) = Health.
# - \(R_i\) = Freedom.
# The centroid is calculated by taking the average of each feature across the six records.
# Formula
# For feature \(j\):
# \[
# \boxed{C_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij}}
# \]

# Where:
# - \(C_j\) = centroid coordinate for feature \(j\).
# - \(n=6\) = number of observations.
# - \(x_{ij}\) = value of feature \(j\) for observation \(i\).
# For four features, the centroid is:
# \[
# C=(\bar E,\bar F,\bar H,\bar R)
# \]

# Example
# Suppose three observations have these two-dimensional coordinates:
# \[
# A=(2,4),\quad B=(4,6),\quad C=(6,8)
# \]

# The centroid is:
# \[
# \bar x=\frac{2+4+6}{3}=4
# \]

# \[
# \bar y=\frac{4+6+8}{3}=6
# \]

# Therefore:
# \[
# \boxed{C=(4,6)}
# \]

# The same calculation applies to four features and six records.
# 3. Load the data and select the first six records
# Python code
# import pandas as pdimport numpy as npdf = pd.read_csv("GHI_Report (1).csv")cols = ["Economy", "Fam", "Health", "Freedom"]data = df.iloc[:6][cols].astype(float)print(data)



# Explanation
# - pd.read_csv() loads the CSV file into a DataFrame.
# - cols lists the four attributes needed.
# - df.iloc[:6] selects the first six rows by position.
# - [cols] keeps only the four specified columns.
# - .astype(float) converts the selected values into floating-point numbers.
# - print(data) displays the six observations.
# Important: iloc[:6] means rows at positions 0 through 5 in Python, which are the first six records. It does not mean rows with index labels 1 through 6.

# The four column names must match the actual CSV headers.
# 4. Calculate the centroid
# Python code
# centroid = data.mean()print("Centroid:")print(centroid)



# Explanation
# - data.mean() calculates the mean of each column.
# - The four means form the centroid.
# - print(centroid) displays the centroid coordinates.
# For example, the output will contain one value each for Economy, Fam, Health, and Freedom. The actual numbers depend on the first six rows of your dataset.
# 5. Euclidean distance
# Euclidean distance is the straight-line distance between two points. It is the most familiar distance measure and is widely used in clustering.
# Formula
# For a point \(X=(x_1,x_2,\ldots,x_p)\) and centroid \(C=(c_1,c_2,\ldots,c_p)\):
# \[
# \boxed{d_E(X,C)=\sqrt{\sum_{j=1}^{p}(x_j-c_j)^2}}
# \]

# For four attributes:
# \[
# d_E=\sqrt{
# (E-\bar E)^2+
# (F-\bar F)^2+
# (H-\bar H)^2+
# (R-\bar R)^2}
# \]

# Example
# Suppose:
# \[
# X=(2,4),\qquad C=(4,6)
# \]

# Then:
# \[
# d_E=\sqrt{(2-4)^2+(4-6)^2}
# \]

# \[
# =\sqrt{4+4}=\sqrt8\approx2.828
# \]

# The distance is approximately 2.828 units.
# Python code
# diff = abs(data - centroid)data["Euclidean"] = np.sqrt((diff ** 2).sum(axis=1))



# Explanation:
# - data - centroid subtracts each centroid coordinate from the corresponding observation coordinate.
# - abs() takes the absolute differences. This is optional for Euclidean distance because the differences are squared.
# - diff ** 2 squares every difference.
# - .sum(axis=1) adds the squared differences across the four features for each row.
# - np.sqrt() takes the square root.
# - The result is stored in the Euclidean column.
# 6. Manhattan distance
# Manhattan distance is the sum of the absolute differences across all features. It is also called city-block distance because it resembles travelling along perpendicular streets rather than moving diagonally.
# Formula
# \[
# \boxed{d_M(X,C)=\sum_{j=1}^{p}|x_j-c_j|}
# \]

# For four attributes:
# \[
# d_M=
# |E-\bar E|+
# |F-\bar F|+
# |H-\bar H|+
# |R-\bar R|
# \]

# Example
# Using the same points:
# \[
# X=(2,4),\qquad C=(4,6)
# \]

# \[
# d_M=|2-4|+|4-6|
# \]

# \[
# =2+2=\boxed4
# \]

# Python code
# data["Manhattan"] = diff.sum(axis=1)



# Explanation:
# - diff contains the absolute difference for each feature.
# - .sum(axis=1) adds those differences across each row.
# - The result is stored in the Manhattan column.
# 7. Minkowski distance with \(p=3\)
# Minkowski distance is a generalised distance measure. Euclidean and Manhattan distances are special cases of this formula.
# General formula
# \[
# \boxed{
# d_p(X,C)=
# \left(\sum_{j=1}^{p_f}|x_j-c_j|^p\right)^{1/p}
# }
# \]

# To avoid confusion, \(p_f\) represents the number of features, while \(p\) is the Minkowski power.

# For this experiment, the Minkowski power is 3 and there are four features:
# \[
# \boxed{
# d_3(X,C)=
# \left(\sum_{j=1}^{4}|x_j-c_j|^3\right)^{1/3}
# }
# \]

# Example
# For the points \(X=(2,4)\) and \(C=(4,6)\):
# \[
# d_3=\left(|2-4|^3+|4-6|^3\right)^{1/3}
# \]

# \[
# =(8+8)^{1/3}
# \]

# \[
# =\sqrt[3]{16}\approx2.520
# \]

# Python code
# data["Minkowski p=3"] = (diff ** 3).sum(axis=1) ** (1/3)



# Explanation:
# - diff ** 3 cubes the absolute differences.
# - .sum(axis=1) adds the cubed differences for each observation.
# - ** (1/3) takes the cube root.
# - The result is stored in the Minkowski p=3 column.
# 8. Comparison of the three distance measures
# Distance	Main operation	Formula
# Euclidean	Square, sum, square root	\(\sqrt{\sum |x_j-c_j|^2}\)
# Manhattan	Absolute differences, sum	\(\sum |x_j-c_j|\)
# Minkowski, \(p=3\)	Cube, sum, cube root	\((\sum |x_j-c_j|^3)^{1/3}\)

# All three distances are zero if the point is exactly at the centroid. Larger values indicate greater distance under the corresponding metric.

# For the same pair of points, Euclidean distance is no greater than Manhattan distance. Minkowski distance with \(p=3\) lies between Euclidean and Manhattan distance:
# \[
# d_E\leq d_3\leq d_M
# \]

# This relationship holds for these standard Minkowski metrics on the same feature differences.

# 9. Pairwise Euclidean distance matrix
# A pairwise distance matrix shows the Euclidean distance between every pair of the six observations.

# For six observations, the matrix is \(6\times6\).
# Formula
# The distance between observations \(X_i\) and \(X_j\) is:
# \[
# \boxed{
# D_{ij}=\sqrt{\sum_{m=1}^{4}(x_{im}-x_{jm})^2}
# }
# \]

# Where:
# - \(D_{ij}\) is the distance between observations \(i\) and \(j\).
# - \(x_{im}\) is the value of feature \(m\) for observation \(i\).
# - \(x_{jm}\) is the value of feature \(m\) for observation \(j\).
# Important properties
# 1. The diagonal values are zero because the distance from a point to itself is zero.
# 2. The matrix is symmetric: \(D_{ij}=D_{ji}\).
# 3. Every distance is non-negative.
# 4. The lower triangular matrix keeps the diagonal and the entries below it.
# For example, the matrix structure is:
# \[
# D=
# \begin{bmatrix}
# 0&0&0&0&0&0\\
# d_{21}&0&0&0&0&0\\
# d_{31}&d_{32}&0&0&0&0\\
# d_{41}&d_{42}&d_{43}&0&0&0\\
# d_{51}&d_{52}&d_{53}&d_{54}&0&0\\
# d_{61}&d_{62}&d_{63}&d_{64}&d_{65}&0
# \end{bmatrix}
# \]

# The symbols \(d_{ij}\) represent the actual distances; the zeros above the diagonal are shown only because we are displaying the lower triangular portion.
# Python code
# from scipy.spatial.distance import cdistX = df.iloc[:6][cols].astype(float).valuesD = cdist(X, X, metric="euclidean")lower = np.tril(D)print(pd.DataFrame(lower, index=data.index, columns=data.index))



# Line-by-line explanation
# Step 1: Select the first six records
# X = df.iloc[:6][cols].astype(float).values



# This selects the four features for the first six observations and converts them into a NumPy array.

# Step 2: Calculate all pairwise distances
# D = cdist(X, X, metric="euclidean")



# - cdist() calculates distances between rows in the first array and rows in the second array.
# - Since both arrays are X, it calculates every observation's distance to every observation.
# - metric="euclidean" selects Euclidean distance.
# - D stores the full symmetric matrix.
# Step 3: Keep the lower triangle
# lower = np.tril(D)



# np.tril() retains the diagonal and all entries below it, replacing the entries above the diagonal with zero.

# Step 4: Display the matrix
# print(pd.DataFrame(lower, index=data.index, columns=data.index))



# This formats the NumPy matrix as a readable pandas DataFrame with row and column labels.

# Note: The pairwise matrix uses the original four features, not distances from the centroid. These are two different calculations.
# 10. Find the nearest and farthest points from the centroid
# After calculating the three distances, we identify which observation has the smallest and largest distance from the centroid.
# Formula
# For each distance measure:
# \[
# \text{Nearest point}=\arg\min_i d_i
# \]

# \[
# \text{Farthest point}=\arg\max_i d_i
# \]

# Where \(d_i\) is the distance of observation \(i\) from the centroid.
# - argmin returns the position or index of the smallest distance.
# - argmax returns the position or index of the largest distance.
# Python code
# dist_cols = ["Euclidean", "Manhattan", "Minkowski p=3"]for col in dist_cols:    nearest = data[col].idxmin()    farthest = data[col].idxmax()    print("\n", col)    print("Nearest point:", nearest, "Distance:", data.loc[nearest, col])    print("Farthest point:", farthest, "Distance:", data.loc[farthest, col])



# Explanation
# 1. dist_cols stores the names of the three distance columns.
# 2. for col in dist_cols: repeats the calculation for each measure.
# 3. data[col].idxmin() returns the index label of the observation with the smallest distance.
# 4. data[col].idxmax() returns the index label of the observation with the largest distance.
# 5. data.loc[nearest, col] retrieves the nearest point's distance.
# 6. data.loc[farthest, col] retrieves the farthest point's distance.
# 7. print() displays the results.
# The nearest and farthest points are identified relative to the centroid, not relative to each other.

# If you want country names instead of index labels, inspect the country-name column in the CSV and use those names as the DataFrame index.
# 11. Important formulas to remember
# Concept	Formula
# Five-number summary	Minimum, Q1, Median, Q3, Maximum
# Interquartile range	\(IQR=Q3-Q1\)
# Lower outlier limit	\(Q1-1.5(IQR)\)
# Upper outlier limit	\(Q3+1.5(IQR)\)
# Centroid	\(\displaystyle C_j=\frac{1}{n}\sum_{i=1}^n x_{ij}\)
# Euclidean distance	\(\displaystyle d_E=\sqrt{\sum_j(x_j-c_j)^2}\)
# Manhattan distance	\(\displaystyle d_M=\sum_j|x_j-c_j|\)
# Minkowski distance, \(p=3\)	\(\displaystyle d_3=(\sum_j|x_j-c_j|^3)^{1/3}\)
# Pairwise Euclidean distance	\(\displaystyle D_{ij}=\sqrt{\sum_m(x_{im}-x_{jm})^2}\)

# 12. Common viva questions and answers
# Q1. What is a box plot?

# A box plot represents the distribution of numerical data using quartiles, the median, whiskers, and potential outliers.

# Q2. What is the five-number summary?

# Minimum, first quartile, median, third quartile, and maximum.

# Q3. What is IQR?

# The interquartile range is the difference between Q3 and Q1. It measures the spread of the middle 50% of observations.

# Q4. Why is 1.5 used in the IQR rule?

# It is a conventional threshold for identifying potential outliers. It is a rule of thumb, not proof that an observation is incorrect.

# Q5. What is a centroid?

# A centroid is the point obtained by calculating the mean of each feature across the observations.

# Q6. What is Euclidean distance?

# It is the straight-line distance between two points.

# Q7. What is Manhattan distance?

# It is the sum of the absolute differences across the features.

# Q8. What is Minkowski distance?

# It is a generalised distance metric. With power 3, the cubed absolute differences are summed and the cube root is taken.

# Q9. Why is the pairwise distance matrix symmetric?

# Because the distance from point A to point B equals the distance from point B to point A.

# Q10. Why are the diagonal entries zero?

# Every observation has zero distance from itself.

# Q11. What does the nearest point from the centroid mean?

# It is the observation with the smallest distance from the mean position of the six observations.

# Q12. What is the difference between the centroid distance and pairwise distance?

# Centroid distance measures the distance from each observation to the mean point. Pairwise distance measures the distance between two individual observations.