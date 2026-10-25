# Data types and visualisation (Exp 1), GHI: Describe the attribute types and dataset

# characteristics (size, dimensionality, missing values, ranges). Draw histograms of all 5 attributes, scatter plots of each factor vs H_Score, a bar chart of the mean factors per Happiness class, and a pie chart of the class counts.

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("GHI_Report (1).csv")

# Dataset description
print("First 5 rows:")
print(df.head())

print("\nDataset size (rows, columns):", df.shape)
print("\nAttribute types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nAttribute ranges:")
print(df.describe())

# Select the 5 numerical attributes
factors = ["Economy", "Fam", "Health", "Freedom"]
attributes = factors + ["H_Score"]

# 1. Histograms of all 5 attributes
df[attributes].hist(figsize=(10, 7), bins=10)
plt.suptitle("Histograms of GHI Attributes")
plt.tight_layout()
plt.show()

# 2. Scatter plots of each factor vs H_Score
for col in factors:
    plt.figure()
    plt.scatter(df[col], df["H_Score"])
    plt.xlabel(col)
    plt.ylabel("H_Score")
    plt.title(col + " vs H_Score")
    plt.show()

# Find the happiness class column
class_col = next(
    (c for c in df.columns if "class" in c.lower()),
    None
)

# If no class column exists, create Low, Medium, High classes
if class_col is None:
    df["Happiness_Class"] = pd.qcut(
        df["H_Score"], q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    )
    class_col = "Happiness_Class"

# 3. Bar chart of mean factors per happiness class
df.groupby(class_col, observed=False)[factors].mean().plot(
    kind="bar", figsize=(8, 5)
)
plt.title("Mean Factors per Happiness Class")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Pie chart of happiness class counts
df[class_col].value_counts().plot(
    kind="pie", autopct="%1.1f%%", figsize=(6, 6)
)
plt.title("Happiness Class Counts")
plt.ylabel("")
plt.show()


# # Experiment 1: Data Types and Visualization — GHI Dataset

# ## 1. Aim of the experiment

# To study the GHI dataset by identifying its attribute types and characteristics, checking missing values and ranges, and visualizing the data using histograms, scatter plots, a bar chart, and a pie chart.

# ## 2. Concepts and definitions

# ### Concept 1: Dataset

# A dataset is a collection of data arranged in rows and columns.

# - Rows: Represent individual observations, such as countries or records.
# - Columns: Represent attributes or features, such as Economy, Health, and Happiness Score.

# For example:

# | Economy | Fam | Health | Freedom | H_Score |
# | ------- | --- | ------ | ------- | ------- |
# | 0.8     | 1.1 | 0.6    | 0.4     | 6.5     |
# | 1.0     | 1.3 | 0.7    | 0.5     | 7.2     |
# | 0.5     | 0.9 | 0.4    | 0.3     | 5.1     |

# These values are illustrative, not actual dataset records.

# ### Concept 2: Attribute types

# An attribute is a characteristic or property recorded in a dataset.

# Common attribute types are:

# - Numerical: Represents a measurable quantity, such as Economy or H_Score.
# - Categorical: Represents a group or label, such as Low, Medium, or High happiness.
# - Discrete: Takes countable values, such as the number of people.
# - Continuous: Can take values within a range, such as a health index or economic score.

# In this experiment, `Economy`, `Fam`, `Health`, `Freedom`, and `H_Score` are expected to be numerical attributes. A happiness class column, if present, is categorical.

# ### Concept 3: Dataset characteristics

# Dataset characteristics describe the structure and quality of the data.

# Size and dimensionality

# - Size tells us how many rows and columns the dataset contains.
# - Dimensionality refers to the number of attributes or features.

# For example, a dataset with 150 rows and 6 columns has a shape of \\((150,6)\\).

# Missing values

# Missing values are entries that have not been recorded. They may appear as blank cells or `NaN`. Missing values can affect calculations and visualizations.

# Attribute ranges

# The range of an attribute is the difference between its maximum and minimum values.

# \\[ \text{Range}=\text{Maximum}-\text{Minimum} \\]

# For example, if the minimum score is 2 and the maximum score is 8:

# \\[ \text{Range}=8-2=6 \\]

# ### Concept 4: Histogram

# A histogram displays the distribution of numerical data by grouping values into intervals called bins.

# It helps us understand:

# - Where most values are concentrated.
# - Whether the distribution is symmetric or skewed.
# - Whether values are spread out or clustered.
# - Whether there may be unusual values.

# The height of each bar represents the number of observations in that interval when frequency is used.

# ### Concept 5: Scatter plot

# A scatter plot displays the relationship between two numerical variables using points.

# In this experiment:

# - X-axis: One factor, such as Economy.
# - Y-axis: `H_Score`.

# We create one scatter plot for each of the four factors.

# Interpretation:

# - Positive relationship: Higher factor values tend to accompany higher H_Score values.
# - Negative relationship: Higher factor values tend to accompany lower H_Score values.
# - No clear relationship: Points do not show a clear upward or downward pattern.

# A scatter plot shows association, not necessarily causation.

# ### Concept 6: Bar chart of mean factors per happiness class

# A bar chart compares values across categories.

# Here, the categories are happiness classes such as Low, Medium, and High. Each bar shows the average value of a factor within a class.

# The formula for the mean is:

# \\[ \bar{x}=\frac{\sum\_{i=1}^{n}x_i}{n} \\]

# Where:

# - \\(x_i\\) is an individual value.
# - \\(n\\) is the number of observations.
# - \\(\bar{x}\\) is the average.

# For example, if Economy values for a class are 0.4, 0.6, and 0.8:

# \\[ \bar{x}=\frac{0.4+0.6+0.8}{3}=0.6 \\]

# The bar chart helps compare average Economy, Fam, Health, and Freedom values between happiness classes.

# ### Concept 7: Pie chart of class counts

# A pie chart displays how a total is divided among categories.

# Each slice represents the proportion of records in one happiness class.

# The percentage formula is:

# \\[ \text{Percentage}= \frac{\text{Class count}}{\text{Total records}}\times100 \\]

# For example, if 30 out of 100 records belong to the Low class:

# \\[ \frac{30}{100}\times100=30\\% \\]

# The Low class occupies 30% of the pie chart.

# ## 3. Explanation of the complete Python code

# ### Step 1: Import libraries

# ```
# import pandas as pdimport matplotlib.pyplot as plt
# ```

# - `pandas` is used to read and analyze tabular data.
# - `matplotlib.pyplot` is used to create charts.
# - `pd` and `plt` are short names for these libraries.

# ### Step 2: Load the dataset

# ```
# df = pd.read_csv("GHI_Report (1).csv")
# ```

# `read_csv()` reads the CSV file and stores it in the DataFrame `df`.

# Make sure the filename matches your actual file.

# ### Step 3: Display the first five rows

# ```
# print("First 5 rows:")print(df.head())
# ```

# `head()` displays the first five records. This helps us check the column names and the format of the data.

# ### Step 4: Check dataset size

# ```
# print("\nDataset size (rows, columns):", df.shape)
# ```

# `shape` returns a pair:

# \\[ (\text{number of rows},\text{number of columns}) \\]

# For example, `(150, 6)` means 150 records and 6 columns.

# ### Step 5: Check attribute types

# ```
# print("\nAttribute types:")print(df.dtypes)
# ```

# `dtypes` displays the data type of every column. Numerical columns may appear as `int64` or `float64`, while text categories often appear as `object` or `string`.

# ### Step 6: Check missing values

# ```
# print("\nMissing values:")print(df.isnull().sum())
# ```

# - `isnull()` identifies missing entries.
# - `sum()` counts them in each column.

# A result of zero means no missing values were detected in that column.

# ### Step 7: Display statistical information and ranges

# ```
# print("\nAttribute ranges:")print(df.describe())
# ```

# `describe()` displays statistics for numerical columns, including:

# - `count`: Number of non-missing values.
# - `mean`: Average value.
# - `std`: Standard deviation.
# - `min`: Minimum value.
# - `25%`: First quartile.
# - `50%`: Median.
# - `75%`: Third quartile.
# - `max`: Maximum value.

# The range can be calculated by subtracting `min` from `max`.

# ### Step 8: Select the five numerical attributes

# ```
# factors = ["Economy", "Fam", "Health", "Freedom"]attributes = factors + ["H_Score"]
# ```

# `factors` contains the four explanatory variables. `attributes` combines these with `H_Score`, giving five columns for visualization.

# ### Step 9: Draw histograms

# ```
# df[attributes].hist(figsize=(10, 7), bins=10)plt.suptitle("Histograms of GHI Attributes")plt.tight_layout()plt.show()
# ```

# - `df[attributes]` selects the five columns.
# - `hist()` creates a histogram for each numerical column.
# - `figsize=(10, 7)` sets the overall figure size.
# - `bins=10` divides each variable's range into 10 intervals.
# - `tight_layout()` adjusts spacing.
# - `show()` displays the charts.

# Result: Five histograms are displayed, one for each numerical attribute.

# ### Step 10: Draw scatter plots

# ```
# for col in factors:    plt.figure()    plt.scatter(df[col], df["H_Score"])    plt.xlabel(col)    plt.ylabel("H_Score")    plt.title(col + " vs H_Score")    plt.show()
# ```

# Explanation:

# - `for col in factors:` repeats the plotting process for Economy, Fam, Health, and Freedom.
# - `plt.figure()` creates a separate figure.
# - `plt.scatter()` plots the factor against `H_Score`.
# - `xlabel()` and `ylabel()` name the axes.
# - `title()` adds a chart title.
# - `show()` displays each chart.

# Result: Four scatter plots are created to examine the relationship between each factor and happiness score.

# ### Step 11: Find or create the happiness class

# ```
# class_col = next(    (c for c in df.columns if "class" in c.lower()),    None)
# ```

# This searches the column names for a name containing `"class"`, regardless of capitalization.

# - If a matching column exists, it is used.
# - If no matching column exists, `class_col` becomes `None`.

# The next part creates classes if necessary:

# ```
# if class_col is None:    df["Happiness_Class"] = pd.qcut(        df["H_Score"], q=3,        labels=["Low", "Medium", "High"],        duplicates="drop"    )    class_col = "Happiness_Class"
# ```

# - `pd.qcut()` divides the observations into approximately equal-sized groups based on their `H_Score` values.
# - `q=3` requests three groups.
# - `labels` names the groups Low, Medium, and High.
# - `duplicates="drop"` allows duplicate bin boundaries to be dropped when necessary.

# Important: If your dataset already has an official happiness class, the code uses it. Otherwise, it creates classes from the score, which may not match the categories defined in your assignment.

# ### Step 12: Draw the bar chart

# ```
# df.groupby(class_col, observed=False)[factors].mean().plot(    kind="bar", figsize=(8, 5))plt.title("Mean Factors per Happiness Class")plt.ylabel("Mean Value")plt.xticks(rotation=0)plt.tight_layout()plt.show()
# ```

# - `groupby(class_col)` groups the records by happiness class.
# - `[factors]` selects the four factors.
# - `.mean()` calculates the average of each factor within each class.
# - `kind="bar"` creates a bar chart.
# - `ylabel()` labels the vertical axis.
# - `xticks(rotation=0)` keeps the category labels horizontal.
# - `show()` displays the chart.

# Result: The chart compares the mean Economy, Fam, Health, and Freedom values for each happiness class.

# ### Step 13: Draw the pie chart

# ```
# df[class_col].value_counts().plot(    kind="pie", autopct="%1.1f%%", figsize=(6, 6))plt.title("Happiness Class Counts")plt.ylabel("")plt.show()
# ```

# - `value_counts()` counts the records in each class.
# - `kind="pie"` creates a pie chart.
# - `autopct="%1.1f%%"` displays percentages to one decimal place.
# - `figsize=(6, 6)` creates a square figure.
# - `ylabel("")` removes the unnecessary vertical-axis label.

# Result: The pie chart shows the percentage of records belonging to each happiness class.

# ## 4. Important formulas for the practical exam

# | Concept                | Formula                                                      |
# | ---------------------- | ------------------------------------------------------------ |
# | Dataset dimensionality | Number of attributes or columns                              |
# | Range                  | Maximum − Minimum                                            |
# | Mean                   | \\(\bar{x}=\frac{\sum x_i}{n}\\)                             |
# | Class percentage       | \\(\frac{\text{Class count}}{\text{Total count}}\times100\\) |

# ## 5. Viva questions and answers

# 1\. What is data visualization?

# Data visualization is the graphical representation of data using charts and plots to understand patterns and relationships.

# 2\. What is an attribute?

# An attribute is a feature or characteristic represented by a column in a dataset.

# 3\. What is a histogram used for?

# It shows the frequency distribution of a numerical variable.

# 4\. What is a scatter plot used for?

# It helps identify relationships between two numerical variables.

# 5\. What does a positive relationship mean?

# As one variable increases, the other tends to increase as well.

# 6\. What is the difference between a bar chart and a histogram?

# A bar chart compares categories, while a histogram shows the distribution of numerical values grouped into intervals.

# 7\. Why is `describe()` used?

# It provides summary statistics such as mean, standard deviation, minimum, maximum, and quartiles.

# 8\. What is a pie chart used for?

# It shows how a total is divided among categories, usually as percentages.

# 9\. What does `isnull().sum()` do?

# It counts the missing values in each column.

# 10\. What does `groupby()` do?

# It divides the data into groups based on a specified column so that calculations can be performed for each group.

# 11\. What does `pd.qcut()` do?

# It divides numerical values into quantile-based groups, aiming to place approximately equal numbers of observations in each group.

# 12\. Why visualize factors against H_Score?

# To observe whether each factor has a positive, negative, or unclear relationship with the happiness score.

# ## 6. Conclusion

# The GHI dataset was examined to understand its size, attribute types, missing values, and numerical ranges. Histograms were used to study the distributions of five numerical attributes. Scatter plots were created to investigate the relationships between four factors and `H_Score`. A bar chart compared the mean factor values across happiness classes, and a pie chart displayed the distribution of class counts. These visualizations help summarize the dataset and identify patterns for further analysis.