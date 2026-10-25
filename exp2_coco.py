# Part A-Box plot analysis (Exp 2), Cocoa: Compute the five-number summary of Rating and of Cocoa % without using describe(). Draw box plots, list the outliers using the IQR rule, and draw side-by-side box plots of Rating for the top 5 company locations.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2).csv")

# Convert Cocoa Percent to numeric
df["Cocoa Percent"] = pd.to_numeric(
    df["Cocoa Percent"].astype(str).str.replace("%", ""),
    errors="coerce"
)

print(df.columns.tolist())

# cell 2
cols = ["Rating", "Cocoa Percent"]

for col in cols:
    x = df[col].dropna()

    print("\n", col)
    print("Minimum:", x.min())
    print("Q1:", x.quantile(0.25))
    print("Median:", x.quantile(0.50))
    print("Q3:", x.quantile(0.75))
    print("Maximum:", x.max())

# cell3
df[cols].plot(kind="box")
plt.title("Box Plots of Rating and Cocoa Percent")
plt.ylabel("Value")
plt.show()
for col in cols:
    x = df[col].dropna()

    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = x[(x < lower) | (x > upper)]

    print("\nOutliers in", col)
    print(outliers.tolist())
top5 = df["Company Location"].value_counts().head(5).index

data = [
    df.loc[df["Company Location"] == loc, "Rating"].dropna()
    for loc in top5
]

plt.boxplot(data, tick_labels=top5)
plt.title("Rating for Top 5 Company Locations")
plt.xlabel("Company Location")
plt.ylabel("Rating")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()    

# 1. Aim of the experiment
# To calculate the five-number summary of Rating and Cocoa Percent without using describe(), draw box plots, identify outliers using the IQR rule, and compare ratings across the five company locations with the most records.
# 2. What is a box plot?
# A box plot, also called a box-and-whisker plot, is a graphical representation of the distribution of numerical data.

# It helps us understand:
# - The centre of the data.
# - The spread or variability of the data.
# - The presence of potential outliers.
# - Whether the data is approximately symmetric or skewed.
# A box plot is based on five key statistics: minimum, Q1, median, Q3, and maximum. In a conventional box plot, the whiskers usually extend to the most extreme observations within the outlier limits, not necessarily the absolute minimum and maximum.







# paytonyau.github.io


# 3. The five-number summary
# The five-number summary describes the distribution using five values.
# Statistic	Meaning
# Minimum	Smallest observed value
# Q1 (first quartile)	25th percentile
# Median (Q2)	50th percentile
# Q3 (third quartile)	75th percentile
# Maximum	Largest observed value

# Understanding quartiles
# Suppose the sorted data is:
# \[
# 2,\ 3,\ 4,\ 5,\ 6,\ 7,\ 8
# \]

# - Minimum = 2
# - Median = 5
# - Q1 = 3.5
# - Q3 = 7
# - Maximum = 8
# Here, Q1 and Q3 follow the percentile interpolation convention used by pandas. Quartiles can differ slightly between software packages because different methods of calculating them exist.
# Python code
# cols = ["Rating", "Cocoa Percent"]for col in cols:    x = df[col].dropna()    print("\n", col)    print("Minimum:", x.min())    print("Q1:", x.quantile(0.25))    print("Median:", x.quantile(0.50))    print("Q3:", x.quantile(0.75))    print("Maximum:", x.max())



# Line-by-line explanation
# 1. Select the columns
# cols = ["Rating", "Cocoa Percent"]



# This creates a list of the two numerical columns being analysed.

# 2. Loop through the columns
# for col in cols:



# This runs the same calculations once for Rating and once for Cocoa Percent.

# 3. Remove missing values
# x = df[col].dropna()



# dropna() removes missing values from the selected column so they do not interfere with the calculations.

# 4. Calculate the minimum
# x.min()



# Returns the smallest valid value.

# 5. Calculate Q1
# x.quantile(0.25)



# Returns the 25th percentile, meaning approximately 25% of the observations are at or below this value.

# 6. Calculate the median
# x.quantile(0.50)



# Returns the 50th percentile, which divides the ordered data into two halves.

# 7. Calculate Q3
# x.quantile(0.75)



# Returns the 75th percentile.

# 8. Calculate the maximum
# x.max()



# Returns the largest valid value.

# The code calculates all five statistics directly without using describe().
# 4. Interquartile Range (IQR)
# The Interquartile Range measures the spread of the middle 50% of the data.
# Formula
# \[
# \boxed{IQR=Q3-Q1}
# \]

# Where:
# - Q1 = first quartile.
# - Q3 = third quartile.
# Example
# Suppose:
# \[
# Q1=3,\qquad Q3=4
# \]

# Then:
# \[
# IQR=4-3=1
# \]

# The middle 50% of the observations lie between 3 and 4.

# A larger IQR indicates that the middle half of the data is more spread out. A smaller IQR indicates that the middle half is more concentrated.
# 5. Outlier detection using the IQR rule
# An outlier is an observation that lies unusually far from the rest of the data.

# The IQR rule uses two limits to identify potential outliers.
# Formulas
# Lower limit:
# \[
# \boxed{L=Q1-1.5(IQR)}
# \]

# Upper limit:
# \[
# \boxed{U=Q3+1.5(IQR)}
# \]

# An observation \(x\) is classified as a potential outlier if:
# \[
# \boxed{x<L\quad\text{or}\quad x>U}
# \]

# The factor 1.5 is a conventional threshold used in box plots. An outlier is not necessarily an error; it may be a genuine observation.
# Example
# Suppose:
# \[
# Q1=3,\quad Q3=4,\quad IQR=1
# \]

# The lower limit is:
# \[
# L=3-1.5(1)=1.5
# \]

# The upper limit is:
# \[
# U=4+1.5(1)=5.5
# \]

# Therefore:
# - A value of 2 is not an outlier.
# - A value of 4.5 is not an outlier.
# - A value of 6 is an outlier because \(6>5.5\).
# - A value of 1 is an outlier because \(1<1.5\).
# Python code
# for col in cols:    x = df[col].dropna()    Q1 = x.quantile(0.25)    Q3 = x.quantile(0.75)    IQR = Q3 - Q1    lower = Q1 - 1.5 * IQR    upper = Q3 + 1.5 * IQR    outliers = x[(x < lower) | (x > upper)]    print("\nOutliers in", col)    print(outliers.tolist())



# Line-by-line explanation
# 1. x = df[col].dropna() selects the column and removes missing values.
# 2. Q1 calculates the first quartile.
# 3. Q3 calculates the third quartile.
# 4. IQR = Q3 - Q1 calculates the interquartile range.
# 5. lower calculates the lower outlier limit.
# 6. upper calculates the upper outlier limit.
# 7. x < lower identifies values below the lower limit.
# 8. x > upper identifies values above the upper limit.
# 9. The | operator means OR, so either condition is sufficient.
# 10. x[(x < lower) | (x > upper)] selects the outlier values.
# 11. outliers.tolist() prints the values as a Python list.
# Important: This code prints the outlier values, not their row numbers. If multiple records share the same outlier value, that value may appear more than once.
# 6. Drawing box plots
# Python code
# df[cols].plot(kind="box")plt.title("Box Plots of Rating and Cocoa Percent")plt.ylabel("Value")plt.show()



# Explanation
# - df[cols] selects the two columns.
# - plot(kind="box") draws a box plot for each column.
# - plt.title() adds the graph title.
# - plt.ylabel() labels the vertical axis.
# - plt.show() displays the graph.
# How to interpret the box plot
# - Line inside the box: Median.
# - Bottom of the box: Q1.
# - Top of the box: Q3.
# - Box height: IQR, representing the middle 50% of observations.
# - Whiskers: Extend to the most extreme observations within the chosen outlier limits.
# - Individual points beyond the whiskers: Potential outliers.
# Note that Rating and Cocoa Percent have different numerical ranges, so the box heights should not be compared as though they were measured on the same scale.
# 7. Side-by-side box plots for the top five company locations
# The purpose is to compare how cocoa ratings vary among the five company locations with the highest numbers of records.
# Python code
# top5 = df["Company Location"].value_counts().head(5).indexdata = [    df.loc[df["Company Location"] == loc, "Rating"].dropna()    for loc in top5]plt.boxplot(data, tick_labels=top5)plt.title("Rating for Top 5 Company Locations")plt.xlabel("Company Location")plt.ylabel("Rating")plt.xticks(rotation=30)plt.tight_layout()plt.show()



# Explanation
# Step 1: Find the top five locations
# top5 = df["Company Location"].value_counts().head(5).index



# - value_counts() counts records for each company location.
# - head(5) selects the five locations with the most records.
# - .index extracts their names.
# Step 2: Collect ratings for each location
# data = [    df.loc[df["Company Location"] == loc, "Rating"].dropna()    for loc in top5]



# For each location, this filters the DataFrame to the matching records and extracts their non-missing ratings. The result is a list of five groups of ratings.

# Step 3: Draw the box plots
# plt.boxplot(data, tick_labels=top5)



# This creates five box plots side by side, one for each location.

# Step 4: Add labels and improve readability
# plt.xlabel("Company Location")plt.ylabel("Rating")plt.xticks(rotation=30)plt.tight_layout()



# These commands label the axes, rotate location names, and reduce overlap.
# How to interpret the graph
# Compare the locations based on:
# - Median: Which location has the highest typical rating?
# - Box height: Which location has more variation in the middle 50% of ratings?
# - Whiskers: How widely are non-outlier ratings spread?
# - Outlier points: Are there unusually high or low ratings?
# The location with the highest median rating is not necessarily the location with the most consistent ratings. Median and spread measure different things.