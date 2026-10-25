# Data preprocessing (Exp 5), GHI: Treat O values in the four factors as missing and impute them with the column median. Detect and cap outliers with the IQR rule, apply Z-score normalisation, and reduce the four factors to 2 principal components (data reduction). Report the explained variance ratio.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("GHI_Report (1).csv")

# Select four factors
cols = ["Economy", "Fam", "Health", "Freedom"]
X = df[cols].copy()

# Replace 0 with missing values
X = X.replace(0, float("nan"))

# Impute missing values with column median
X = X.fillna(X.median())

# Detect and cap outliers using IQR
for col in cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((X[col] < lower) | (X[col] > upper)).sum()
    print(col, "Outliers:", outliers)

    X[col] = X[col].clip(lower, upper)

# Apply Z-score normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Display explained variance ratio
print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal explained variance:")
print(pca.explained_variance_ratio_.sum())

# Display first 5 rows of reduced data
print("\nFirst 5 rows of PCA data:")
print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).head())

# Display final reduced dataset
print("\nReduced data shape:", X_pca.shape)

# # Experiment 5: Data Preprocessing — Complete Explanation, Formulas and Concepts

# This experiment uses the GHI (World Happiness) dataset and the four factors `Economy`, `Fam`, `Health`, and `Freedom`.

# The main objective is to clean the data, handle missing values, detect and cap outliers, standardize the values, and reduce the four features to two principal components using PCA.

# ## 1. Aim of the experiment

# To perform the following data preprocessing operations:

# 1. Treat zero values in the four selected factors as missing values.
# 2. Replace missing values using the median of each column.
# 3. Detect and cap outliers using the Interquartile Range (IQR) method.
# 4. Apply Z-score normalization.
# 5. Reduce four features to two principal components using Principal Component Analysis (PCA).
# 6. Calculate and interpret the explained variance ratio.

# ## 2. Concepts and formulas

# ### Concept 1: Data preprocessing

# Data preprocessing is the process of preparing raw data so it can be analyzed or used in machine-learning algorithms.

# Real-world datasets may contain missing values, unusual observations, different measurement scales, or unnecessary features. These issues can affect the accuracy and reliability of analysis.

# In this experiment, preprocessing involves missing-value treatment, outlier handling, standardization, and dimensionality reduction.

# ### Concept 2: Treating zero values as missing

# Sometimes, a value of `0` means that a measurement was not recorded rather than representing a genuine zero.

# For example, if a country's health score is recorded as zero because the value is missing, treating that zero as a real score could distort the analysis.

# The code is:

# ```
# X = X.replace(0, float("nan"))
# ```

# Explanation:

# - `X` contains the four selected columns.
# - `replace(0, float("nan"))` replaces every zero in those columns with `NaN`.
# - `NaN` means “Not a Number” and is commonly used to represent missing numerical data.

# Important: This is appropriate only if the experiment explicitly says that zero values should be treated as missing. If zero is a valid measurement, it should not automatically be replaced.

# ### Concept 3: Missing-value imputation using the median

# Imputation means replacing missing values with estimated or representative values so that the data can be processed.

# The median is the middle value when observations are arranged in ascending order.

# For sorted values \\(x_1,x_2,\ldots,x_n\\):

# \\[ \operatorname{Median}(X)= \begin{cases} x\_{\frac{n+1}{2}}, & n\text{ odd}\\\\[6pt] \frac{x\_{\frac n2}+x\_{\frac n2+1}}{2}, & n\text{ even} \end{cases} \\]

# For an even number of observations, the median is the average of the two middle values.

# Example: Consider the values

# \\[ 10,\ 12,\ \text{missing},\ 14,\ 100 \\]

# Ignoring the missing value, the observed values are \\(10,12,14,100\\).

# \\[ \text{Median}=\frac{12+14}{2}=13 \\]

# The missing value is replaced with 13.

# The code is:

# ```
# X = X.fillna(X.median())
# ```

# Explanation:

# - `X.median()` calculates the median of each column separately.
# - `fillna()` replaces missing values with the corresponding column median.
# - Each feature gets its own median.

# Why use the median? The median is less affected by extreme values than the mean. For example, the mean of \\(10,12,14,100\\) is 34, while the median is 13.

# ### Concept 4: Outlier detection using IQR

# An outlier is an observation that lies unusually far from the majority of the data.

# The IQR method detects potential outliers by examining the spread of the middle 50% of the observations.

# The required formulas are:

# First quartile (\\(Q_1\\)): The 25th percentile.

# Third quartile (\\(Q_3\\)): The 75th percentile.

# Interquartile range:

# \\[ IQR=Q_3-Q_1 \\]

# Lower bound:

# \\[ L=Q_1-1.5(IQR) \\]

# Upper bound:

# \\[ U=Q_3+1.5(IQR) \\]

# A value \\(x\\) is flagged as an outlier if:

# \\[ x\<L\quad\text{or}\quad x>U \\]

# The factor 1.5 is a conventional rule for identifying potential outliers; it does not mean every flagged value is an error.

# Example: Suppose a column has:

# \\[ Q_1=20,\qquad Q_3=40 \\]

# Then:

# \\[ IQR=40-20=20 \\]

# \\[ L=20-1.5(20)=-10 \\]

# \\[ U=40+1.5(20)=70 \\]

# Any value below \\(-10\\) or above \\(70\\) is flagged as an outlier. A value of 90 is therefore an outlier according to this rule.

# ### Concept 5: Outlier capping

# After detecting outliers, this experiment caps them rather than deleting the rows.

# Capping replaces values outside the calculated bounds with the nearest bound:

# \\[ x\_{\text{capped}}= \begin{cases} L, & x\<L\\\ x, & L\leq x\leq U\\\ U, & x>U \end{cases} \\]

# For the example above, a value of 90 becomes 70, while a value of 35 remains unchanged.

# This technique is also called winsorization when values are replaced by specified percentile-based limits.

# The code is:

# ```
# for col in cols:    Q1 = X[col].quantile(0.25)    Q3 = X[col].quantile(0.75)    IQR = Q3 - Q1    lower = Q1 - 1.5 * IQR    upper = Q3 + 1.5 * IQR    outliers = ((X[col] < lower) | (X[col] > upper)).sum()    print(col, "Outliers:", outliers)    X[col] = X[col].clip(lower, upper)
# ```

# Line-by-line explanation:

# - `for col in cols:` repeats the process for each of the four features.
# - `quantile(0.25)` calculates the first quartile.
# - `quantile(0.75)` calculates the third quartile.
# - `IQR = Q3 - Q1` calculates the interquartile range.
# - `lower` and `upper` calculate the lower and upper outlier bounds.
# - The Boolean expression checks which values fall outside those bounds.
# - `.sum()` counts the flagged observations because `True` is counted as 1.
# - `.clip(lower, upper)` caps values at the bounds.

# The printed outlier counts are calculated before capping. They indicate how many observations were flagged in each column at that stage.

# Why cap rather than remove outliers? Capping retains all rows while limiting the influence of extreme values on later analysis.

# ### Concept 6: Z-score normalization (standardization)

# Z-score normalization transforms numerical values so that each feature has approximately:

# - Mean = 0
# - Standard deviation = 1

# This is important because the four features may have different ranges or amounts of variation. PCA is sensitive to scale, so standardizing the features helps prevent a feature with a larger numerical spread from dominating the analysis.

# Formula:

# \\[ z=\frac{x-\mu}{\sigma} \\]

# Where:

# - \\(x\\) = original value
# - \\(\mu\\) = mean of the feature
# - \\(\sigma\\) = standard deviation of the feature
# - \\(z\\) = standardized value

# Example: Suppose the mean of a feature is 50, its standard deviation is 10, and one observation is 70.

# \\[ z=\frac{70-50}{10}=2 \\]

# The standardized value is 2, meaning the observation is two standard deviations above the mean.

# Another observation of 40 would have:

# \\[ z=\frac{40-50}{10}=-1 \\]

# It is one standard deviation below the mean.

# The code is:

# ```
# scaler = StandardScaler()X_scaled = scaler.fit_transform(X)
# ```

# Explanation:

# - `StandardScaler()` creates a standardization object.
# - `fit_transform(X)` calculates each feature's mean and standard deviation, then transforms the data.
# - `X_scaled` contains the standardized values as a NumPy array.

# Scikit-learn's `StandardScaler` uses the population-style standard deviation, calculated with a divisor of \\(n\\), rather than the sample standard deviation with divisor \\(n-1\\).

# Why is Z-score normalization needed before PCA?

# PCA identifies directions of maximum variance. If one feature has a much larger scale or variance than another, it can disproportionately influence those directions. Standardization makes the features more comparable before PCA.

# ### Concept 7: Principal Component Analysis (PCA)

# Principal Component Analysis (PCA) is a dimensionality-reduction technique. It transforms a set of potentially correlated features into a smaller set of new features called principal components.

# In this experiment, the original data contains four features:

# - Economy
# - Fam
# - Health
# - Freedom

# PCA reduces these four features to two:

# - PC1: First principal component
# - PC2: Second principal component

# Each principal component is a weighted combination of the original standardized features.

# For example, the first component has the general form:

# \\[ PC_1=a_1z_1+a_2z_2+a_3z_3+a_4z_4 \\]

# The second component has the form:

# \\[ PC_2=b_1z_1+b_2z_2+b_3z_3+b_4z_4 \\]

# Here, \\(z_1,z_2,z_3,z_4\\) represent the standardized feature values, while the coefficients \\(a_i\\) and \\(b_i\\) are weights learned by PCA.

# The first component captures as much variance as possible in one direction. The second captures as much of the remaining variance as possible while being orthogonal (uncorrelated in the training data) to the first component.

# How PCA works conceptually:

# Original dataset

# 4 features: Economy, Fam, Health, Freedom

# Standardized data

# Each feature has mean ≈ 0 and standard deviation ≈ 1

# PCA transformation

# Finds directions that capture the most variance

# Reduced dataset

# 2 features: PC1 and PC2

# The code is:

# ```
# pca = PCA(n_components=2)X_pca = pca.fit_transform(X_scaled)
# ```

# Explanation:

# - `PCA(n_components=2)` instructs PCA to retain two principal components.
# - `fit_transform(X_scaled)` learns the components from the standardized data and projects each observation onto them.
# - `X_pca` stores the transformed data.

# If the original dataset has \\(n\\) rows, the shape changes from:

# \\[ (n,4)\longrightarrow(n,2) \\]

# The number of observations remains the same, but the number of features is reduced from four to two.

# Why use PCA?

# - It reduces the number of features.
# - It can simplify analysis and visualization.
# - It may remove redundant information when features are correlated.
# - It can reduce computational complexity in later modelling.

# Important limitation: PCA components are combinations of the original features, so PC1 and PC2 are not automatically equivalent to individual factors such as Economy or Health. Some interpretability is lost.

# ### Concept 8: Explained variance ratio

# The explained variance ratio measures how much of the total variance in the standardized dataset is captured by each principal component.

# For component \\(i\\):

# \\[ EVR_i=\frac{\lambda_i}{\sum\_{j=1}^{p}\lambda_j} \\]

# Where:

# - \\(\lambda_i\\) = variance captured by principal component \\(i\\), represented by its eigenvalue
# - \\(p\\) = number of original features
# - \\(\sum\_{j=1}^{p}\lambda_j\\) = total variance across all components

# For this experiment, there are four standardized features and two retained components.

# The code is:

# ```
# print("\nExplained variance ratio:")print(pca.explained_variance_ratio_)print("\nTotal explained variance:")print(pca.explained_variance_ratio_.sum())
# ```

# The first line prints the proportion of variance explained by PC1 and PC2 separately. The second adds those proportions together to show the total variance retained by the two-component representation.

# Example using hypothetical values:

# Suppose the output is:

# ```
# Explained variance ratio:
# [0.55 0.25]

# Total explained variance:
# 0.80
# ```

# Then:

# - PC1 explains 55% of the variance.
# - PC2 explains 25% of the variance.
# - Together, they explain 80% of the variance.

# \\[ 55\\%+25\\%=80\\% \\]

# This means the two components retain 80% of the variance represented in the four standardized features, while 20% is not captured by these two components.

# These are only example values. Your actual explained variance ratio depends on the dataset and must be read from your program's output.

# How to interpret the result:

# - A higher cumulative explained variance means the two components retain more of the original dataset's variation.
# - A lower cumulative explained variance means more variation is omitted by reducing to two components.
# - There is no universal percentage that guarantees PCA is good; the acceptable amount depends on the task and dataset.

# The final code:

# ```
# print("\nFirst 5 rows of PCA data:")print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).head())print("\nReduced data shape:", X_pca.shape)
# ```

# creates a DataFrame with the names `PC1` and `PC2`, displays the first five transformed observations, and prints the dimensions of the reduced data.

# The reduced data contains component scores, not the original factor values.

# ## 3. Complete code explained step by step

# Here is the complete program for reference.

# ```
# import pandas as pdfrom sklearn.preprocessing import StandardScalerfrom sklearn.decomposition import PCA# Load datasetdf = pd.read_csv("GHI_Report (1).csv")# Select four factorscols = ["Economy", "Fam", "Health", "Freedom"]X = df[cols].copy()# Replace 0 with missing valuesX = X.replace(0, float("nan"))# Impute missing values with column medianX = X.fillna(X.median())# Detect and cap outliers using IQRfor col in cols:    Q1 = X[col].quantile(0.25)    Q3 = X[col].quantile(0.75)    IQR = Q3 - Q1    lower = Q1 - 1.5 * IQR    upper = Q3 + 1.5 * IQR    outliers = ((X[col] < lower) | (X[col] > upper)).sum()    print(col, "Outliers:", outliers)    X[col] = X[col].clip(lower, upper)# Apply Z-score normalisationscaler = StandardScaler()X_scaled = scaler.fit_transform(X)# Reduce to 2 principal componentspca = PCA(n_components=2)X_pca = pca.fit_transform(X_scaled)# Display explained variance ratioprint("\nExplained variance ratio:")print(pca.explained_variance_ratio_)print("\nTotal explained variance:")print(pca.explained_variance_ratio_.sum())# Display first 5 rows of reduced dataprint("\nFirst 5 rows of PCA data:")print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).head())# Display final reduced datasetprint("\nReduced data shape:", X_pca.shape)
# ```

# ### Explanation of the first lines

# ```
# import pandas as pd
# ```

# Imports Pandas, which is used to read CSV files and manipulate tabular data. The alias `pd` makes it easier to write Pandas commands.

# ```
# from sklearn.preprocessing import StandardScaler
# ```

# Imports the class used to standardize the features with the Z-score formula.

# ```
# from sklearn.decomposition import PCA
# ```

# Imports the PCA algorithm used for dimensionality reduction.

# ```
# df = pd.read_csv("GHI_Report (1).csv")
# ```

# Reads the CSV file into a Pandas DataFrame named `df`. The filename must match the name of your actual file.

# ```
# cols = ["Economy", "Fam", "Health", "Freedom"]X = df[cols].copy()
# ```

# Selects only the four required columns. The `.copy()` creates a separate working copy so that the preprocessing steps do not unintentionally modify the selected columns in the original DataFrame.

# ### Explanation of the final output

# | Output                     | Meaning                                                         |
# | -------------------------- | --------------------------------------------------------------- |
# | `Outliers:`                | Number of values flagged in each feature before capping         |
# | `Explained variance ratio` | Proportion of variance explained by PC1 and PC2 individually    |
# | `Total explained variance` | Combined variance explained by the two components               |
# | `First 5 rows of PCA data` | First five observations represented by their PC1 and PC2 scores |
# | `Reduced data shape`       | Number of rows and retained components                          |

# For example, if the dataset has 150 observations, the reduced data shape will be:

# ```
# Reduced data shape: (150, 2)
# ```

# This indicates 150 observations and two principal components. Your actual row count may differ.

# ## 4. Summary of all important formulas

# | Operation                     | Formula                                       |
# | ----------------------------- | --------------------------------------------- |
# | Median, odd \\(n\\)           | Middle sorted observation                     |
# | Median, even \\(n\\)          | Average of the two middle observations        |
# | Interquartile range           | \\(IQR=Q_3-Q_1\\)                             |
# | Lower outlier bound           | \\(Q_1-1.5(IQR)\\)                            |
# | Upper outlier bound           | \\(Q_3+1.5(IQR)\\)                            |
# | Z-score                       | \\(z=(x-\mu)/\sigma\\)                        |
# | Explained variance ratio      | \\(EVR_i=\lambda_i/\sum_j\lambda_j\\)         |
# | Cumulative explained variance | \\(EVR_1+EVR_2\\) for two retained components |

# ## 5. Advantages and limitations

# Advantages

# - Median imputation fills missing values without removing rows.
# - IQR-based capping reduces the influence of extreme observations.
# - Z-score standardization puts features on a comparable scale.
# - PCA reduces four features to two and makes visualization easier.
# - Explained variance helps measure how much information is retained.

# Limitations

# - Replacing zero values is only correct if zero truly represents missing data.
# - Median imputation can reduce natural variation and does not account for why values are missing.
# - IQR may flag legitimate extreme observations as outliers.
# - Standardization does not make the data normally distributed.
# - PCA can make the original features harder to interpret and does not guarantee better model performance.

# ## 6. Viva questions and answers

# 1\. What is data preprocessing?

# Data preprocessing is the process of cleaning and transforming raw data before analysis or machine learning.

# 2\. Why are zero values replaced with missing values?

# Because the experiment assumes that zero represents missing data for the four selected factors.

# 3\. Why is median imputation used?

# The median is less affected by extreme values than the mean.

# 4\. What is an outlier?

# An observation that lies unusually far from the main distribution of the data.

# 5\. What is IQR?

# The interquartile range is the difference between the third and first quartiles: \\(Q_3-Q_1\\).

# 6\. Why is the multiplier 1.5 used in the IQR method?

# It is a conventional rule for identifying potential outliers beyond the quartile-based limits.

# 7\. What is Z-score normalization?

# It transforms values by subtracting the feature mean and dividing by its standard deviation.

# 8\. Why do we standardize before PCA?

# To prevent differences in feature scales from disproportionately influencing the variance-based PCA calculation.

# 9\. What is PCA?

# PCA is a dimensionality-reduction method that transforms the original features into principal components that capture as much variance as possible.

# 10\. What are PC1 and PC2?

# PC1 captures the greatest possible variance in one direction; PC2 captures the greatest possible remaining variance in a direction orthogonal to PC1.

# 11\. What is explained variance ratio?

# It is the proportion of total variance captured by a principal component.

# 12\. Why reduce four features to two?

# To simplify the dataset and make it easier to visualize while retaining as much variation as possible.

# 13\. Does PCA remove rows?

# No. In this experiment, PCA retains the observations but transforms the four selected features into two components.

# 14\. What does an explained variance ratio of 0.80 mean for two components?

# The two retained components together capture 80% of the variance in the standardized input data.

# 15\. What is the difference between outlier detection and outlier capping?

# Detection identifies unusual values; capping replaces values outside the bounds with the corresponding lower or upper limit.

# ## 7. Conclusion for your practical record

# The GHI dataset was preprocessed by treating zero values as missing, imputing missing values using column medians, detecting and capping outliers using the IQR method, and standardizing the four selected features using Z-score normalization. PCA was then applied to reduce the four features to two principal components. The explained variance ratios of PC1 and PC2 were calculated to evaluate the proportion of the original standardized data's variance retained after dimensionality reduction.

# Remember: Write the actual explained variance ratio and cumulative percentage from your program's output in the final practical record rather than using the hypothetical example.