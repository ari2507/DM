```python
import pandas as pd

# 1. Load dataset
df = pd.read_csv("GHI_Report (1).csv")
df.columns = df.columns.str.strip()

# 2. Select required columns
factors = ["Economy", "Fam", "Health", "Freedom"]
df["H_Score"] = pd.to_numeric(df["H_Score"], errors="coerce")

for col in factors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["H_Score"] + factors).copy()

# 3. Generalise H_Score using fixed cut-offs
def happiness_level(score):
    if score >= 6.5:
        return "Very High"
    elif score >= 5.5:
        return "High"
    elif score >= 4.5:
        return "Medium"
    else:
        return "Low"

df["H_Score_Level"] = df["H_Score"].apply(happiness_level)

# 4. Generalise Economy using tertiles
df["Economy_Level"] = pd.qcut(
    df["Economy"].rank(method="first"),
    q=3,
    labels=["Poor", "Middle", "Rich"]
)

# 5. Display concept hierarchies
print("H_Score concept hierarchy:")
print(df[["H_Score", "H_Score_Level"]].head(10))

print("\nEconomy concept hierarchy:")
print(df[["Economy", "Economy_Level"]].head(10))

# 6. Mean of each factor per H_Score level
print("\nMean of each factor per H_Score level:")
mean_table = df.groupby("H_Score_Level", observed=False)[factors].mean()
print(mean_table.round(3))

# 7. Economy Level vs H_Score Level count table
print("\nEconomy Level x H_Score Level count table:")
count_table = pd.crosstab(
    df["Economy_Level"],
    df["H_Score_Level"]
)

# Keep rows and columns in a logical order
count_table = count_table.reindex(
    index=["Poor", "Middle", "Rich"],
    columns=["Low", "Medium", "High", "Very High"],
    fill_value=0
)

print(count_table)

# 8. Display number of records in each generalised class
print("\nH_Score level counts:")
print(df["H_Score_Level"].value_counts())

print("\nEconomy level counts:")
print(df["Economy_Level"].value_counts().reindex(
    ["Poor", "Middle", "Rich"], fill_value=0
))
```

# Explanation of the code
# 1. What is data generalisation?
# Data generalisation replaces detailed numerical values with broader categories. It makes data easier to summarize and understand.

# For example, instead of displaying an exact happiness score of 6.8, we can display its generalised category, Very High.

# This process is also called creating a concept hierarchy, where detailed values are mapped to higher-level concepts.
# 2. H_Score concept hierarchy
# The experiment specifies three fixed cut-offs: 6.5, 5.5, and 4.5.
# H_Score range	Generalised level
# \(H\_Score \geq 6.5\)	Very High
# \(5.5 \leq H\_Score < 6.5\)	High
# \(4.5 \leq H\_Score < 5.5\)	Medium
# \(H\_Score < 4.5\)	Low

# The code uses:
# def happiness_level(score):    if score >= 6.5:        return "Very High"    elif score >= 5.5:        return "High"    elif score >= 4.5:        return "Medium"    else:        return "Low"



# The function checks the score against each threshold and returns the corresponding category. The apply() function applies this rule to every row.
# 3. Economy concept hierarchy using tertiles
# Tertiles divide sorted numerical observations into three approximately equal-sized groups.

# The formula for the two tertile cut-offs is based on the 33.33rd and 66.67th percentiles:
# \[
# T_1=P_{33.33}, \qquad T_2=P_{66.67}
# \]

# The categories are:
# - Poor: Lower third of Economy values.
# - Middle: Middle third.
# - Rich: Upper third.
# The code uses:
# df["Economy_Level"] = pd.qcut(    df["Economy"].rank(method="first"),    q=3,    labels=["Poor", "Middle", "Rich"])



# rank(method="first") gives tied values distinct ranks according to their order in the dataset. This helps qcut() form three groups even when Economy contains repeated values. Consequently, identical Economy values can fall into different groups if they occur around a tertile boundary.
# 4. Mean of each factor per happiness level
# The code is:
# mean_table = df.groupby("H_Score_Level", observed=False)[factors].mean()



# It groups the observations by happiness level and calculates the mean of Economy, Fam, Health, and Freedom in each group.

# The formula is:
# \[
# \bar{x}=\frac{\sum_{i=1}^{n}x_i}{n}
# \]

# For example, if the Economy values in a group are 0.4, 0.6, and 0.8:
# \[
# \bar{x}=\frac{0.4+0.6+0.8}{3}=0.6
# \]

# The result helps compare the average factor values across Low, Medium, High, and Very High happiness groups.
# 5. Economy level versus H_Score level count table
# The code is:
# count_table = pd.crosstab(    df["Economy_Level"],    df["H_Score_Level"])



# A cross-tabulation counts how many observations fall into every combination of two categories.

# For example, a table cell at row Rich and column Very High gives the number of records that are both in the Rich Economy group and the Very High happiness group.

# This table can reveal whether higher Economy groups tend to contain more high-happiness observations.
# 6. Why display class counts?
# The final two outputs count the records in each generalised class. They help verify that categories were created and show how the observations are distributed.

# Economy tertiles should contain approximately equal numbers of observations. Happiness classes may have very different counts because their cut-offs are fixed rather than chosen to create equally sized groups.
# Expected output
# The program prints:
# 1. The first 10 original H_Score values and their generalised levels.
# 2. The first 10 Economy values and their Economy levels.
# 3. A table of mean Economy, Fam, Health, and Freedom for each happiness level.
# 4. An Economy-level-by-Happiness-level count table.
# 5. Counts for each category.
# Exact values depend on your CSV file.
# Viva questions and answers
# 1. What is data generalisation?

# It replaces detailed data values with broader, more understandable categories.

# 2. What is a concept hierarchy?

# A hierarchy that maps detailed values to higher-level concepts, such as numerical scores to Low, Medium, High, and Very High.

# 3. What are tertiles?

# Tertiles divide ordered observations into three approximately equal-sized groups.

# 4. Why are fixed cut-offs used for H_Score?

# They follow the experiment's specified thresholds and make the happiness categories consistent.

# 5. What is the purpose of pd.qcut()?

# It divides data into quantile-based groups with approximately equal numbers of observations.

# 6. What does groupby().mean() do?

# It groups the data by a category and calculates the average of selected numerical columns in each group.

# 7. What is a cross-tabulation?

# It counts observations for combinations of categories from two variables.

# 8. Why might the Economy tertile groups not have exactly the same size?

# Tertile groups may differ slightly in size due to the number of observations and the handling of ties.

# 9. Does association mean causation?

# No. A relationship between Economy level and happiness level does not establish that Economy alone causes happiness.
# Conclusion
# The GHI dataset was generalised by converting H_Score into four levels using the specified fixed cut-offs and Economy into Poor, Middle, and Rich levels using tertiles. The mean values of the four factors were calculated for each happiness level, and a cross-tabulation was generated to compare Economy levels with happiness levels. These summaries help reveal patterns between economic conditions and happiness in the dataset.