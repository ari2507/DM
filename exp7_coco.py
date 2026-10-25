```python
import pandas as pd
from itertools import combinations
from collections import Counter

# 1. Load the Cocoa dataset
df = pd.read_csv("flavors_of_cocoa (1) (2).csv")

# Clean column names
df.columns = df.columns.str.strip()

# 2. Prepare the required columns
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df["Cocoa Percent"] = (
    df["Cocoa Percent"].astype(str).str.replace("%", "", regex=False)
)
df["Cocoa Percent"] = pd.to_numeric(df["Cocoa Percent"], errors="coerce")

df["Review Date"] = pd.to_numeric(df["Review Date"], errors="coerce")

# Remove rows missing any required value
df = df.dropna(
    subset=["Company Location", "Cocoa Percent", "Rating", "Review Date"]
).copy()

df["Review Date"] = df["Review Date"].astype(int)

# Keep the specified review periods
df = df[df["Review Date"].between(2006, 2017)].copy()

# 3. Create Company Location items
top_locations = df["Company Location"].value_counts().head(8).index

df["Location_Item"] = df["Company Location"].where(
    df["Company Location"].isin(top_locations), "Other"
)
df["Location_Item"] = "Location-" + df["Location_Item"].astype(str)

# 4. Create Cocoa bands
df["Cocoa_Item"] = df["Cocoa Percent"].apply(
    lambda x: "Cocoa<70" if x < 70 else "Cocoa>=70"
)

# 5. Create Rating classes
# Low: below 3; Medium: 3 to below 4; High: 4 or above
def rating_class(rating):
    if rating < 3:
        return "Rating-Low"
    elif rating < 4:
        return "Rating-Medium"
    else:
        return "Rating-High"

df["Rating_Item"] = df["Rating"].apply(rating_class)

# 6. Create Review Period items
def review_period(year):
    if 2006 <= year <= 2009:
        return "Period-2006-09"
    elif 2010 <= year <= 2013:
        return "Period-2010-13"
    else:
        return "Period-2014-17"

df["Period_Item"] = df["Review Date"].apply(review_period)

# 7. Create one transaction per review, with 4 items
transactions = [
    frozenset(row)
    for row in df[
        ["Location_Item", "Cocoa_Item", "Rating_Item", "Period_Item"]
    ].values
]

print("Number of transactions:", len(transactions))
print("Example transactions:")
for t in transactions[:5]:
    print(sorted(t))

# 8. Apriori algorithm implemented from scratch
min_support = 0.05
n = len(transactions)

def get_support(itemset):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / n

# Generate L1
all_items = sorted(set(item for t in transactions for item in t))
L1 = {
    frozenset([item]): get_support(frozenset([item]))
    for item in all_items
    if get_support(frozenset([item])) >= min_support
}

# Generate candidate itemsets from the previous frequent level
def generate_candidates(previous_level, k):
    candidates = set()
    items = sorted(set().union(*previous_level.keys()))

    for combo in combinations(items, k):
        candidate = frozenset(combo)

        # Apriori pruning: every (k-1)-subset must be frequent
        if all(
            frozenset(subset) in previous_level
            for subset in combinations(combo, k - 1)
        ):
            candidates.add(candidate)

    return candidates

# Generate L2 and L3
frequent_levels = {1: L1}

for k in [2, 3]:
    candidates = generate_candidates(frequent_levels[k - 1], k)

    current_level = {}
    for candidate in candidates:
        support = get_support(candidate)
        if support >= min_support:
            current_level[candidate] = support

    frequent_levels[k] = current_level

# 9. Print L1, L2 and L3 with support
def print_level(level_name, itemsets):
    print("\n" + level_name)
    if not itemsets:
        print("No frequent itemsets found.")
    else:
        for itemset, support in sorted(
            itemsets.items(),
            key=lambda x: (len(x[0]), sorted(x[0]))
        ):
            print(sorted(itemset), "Support:", round(support, 4))

print_level("L1 (Frequent 1-itemsets)", frequent_levels[1])
print_level("L2 (Frequent 2-itemsets)", frequent_levels[2])
print_level("L3 (Frequent 3-itemsets)", frequent_levels[3])

# 10. Verify support by directly counting 3 itemsets
all_frequent = []
for level in [1, 2, 3]:
    all_frequent.extend(frequent_levels[level].keys())

print("\nDirect support verification:")
for itemset in all_frequent[:3]:
    direct_count = sum(
        1 for t in transactions if itemset.issubset(t)
    )
    direct_support = direct_count / n

    print("Itemset:", sorted(itemset))
    print("Direct count:", direct_count)
    print("Direct support:", round(direct_support, 4))
    print("Apriori support:",
          round(frequent_levels[len(itemset)][itemset], 4))
    print()
```

# Explanation of the code
# 1. What is the Apriori algorithm?
# The Apriori algorithm is a data-mining algorithm used to discover frequent itemsets in transaction data.

# For example, a cocoa review may be represented by four items:
# - Location-Canada
# - Cocoa<70
# - Rating-High
# - Period-2014-17
# Apriori identifies which combinations of items appear frequently across reviews.
# 2. Concepts and formulas
# Support

# Support measures how frequently an itemset occurs in all transactions.
# \[
# \text{Support}(A)=
# \frac{\text{Transactions containing }A}
# {\text{Total transactions}}
# \]

# Suppose 100 reviews are processed and 12 contain both Cocoa<70 and Rating-High.
# \[
# \text{Support}=\frac{12}{100}=0.12=12\%
# \]

# Because the minimum support is 0.05, this itemset qualifies as frequent.

# Minimum support
# min_support = 0.05



# This means an itemset must appear in at least 5% of all transactions to be considered frequent.

# For 1,000 transactions, an itemset must occur in at least 50 transactions.

# Frequent itemsets
# - \(L_1\): Frequent itemsets containing one item.
# - \(L_2\): Frequent itemsets containing two items.
# - \(L_3\): Frequent itemsets containing three items.
# For example:
# Level	Example itemset
# \(L_1\)	{Rating-High}
# \(L_2\)	{Rating-High, Cocoa<70}
# \(L_3\)	{Rating-High, Cocoa<70, Location-Canada}

# These examples are illustrative; your actual frequent itemsets depend on the dataset.

# Apriori principle

# The key principle is:
# If an itemset is frequent, all its non-empty subsets must also be frequent.

# For example, if {A, B, C} is frequent, then {A, B}, {A, C}, and {B, C} must also be frequent.

# If {A, B} is not frequent, there is no need to test {A, B, C}. This reduces the number of candidate itemsets that must be checked.
# 3. Explanation of the main code sections
# Code section	Purpose
# Load CSV	Reads the Cocoa dataset
# Clean columns and values	Converts Rating, Cocoa Percent, and Review Date to numerical values
# Top 8 locations	Keeps the eight most common locations and groups the rest as Other
# Cocoa band	Categorizes values below 70% and values of 70% or above
# Rating class	Categorizes ratings into Low, Medium, and High
# Review period	Assigns each review to its specified time period
# Transactions	Creates one set of four items for every review
# get_support()	Counts transactions containing an itemset and calculates support
# generate_candidates()	Creates larger candidates and prunes those with infrequent subsets
# frequent_levels	Stores \(L_1\), \(L_2\), and \(L_3\)
# Direct verification	Recounts selected itemsets directly and compares their supports

# 4. Why are transactions stored as sets?
# The code uses:
# transactions = [    frozenset(row)    for row in df[        ["Location_Item", "Cocoa_Item", "Rating_Item", "Period_Item"]    ].values]



# A set represents the items in a review. A frozenset is an immutable set, which can also be used as a dictionary key when storing itemsets and their support values.

# Each transaction contains four distinct item categories: location, cocoa band, rating class, and review period.
# 5. How is support calculated in the code?
# def get_support(itemset):    count = sum(1 for t in transactions if itemset.issubset(t))    return count / n



# - itemset.issubset(t) checks whether all items in the candidate are present in a transaction.
# - sum(...) counts how many transactions contain the complete itemset.
# - Dividing by n, the total number of transactions, gives its support.
# 6. How is support verified directly?
# The code counts the transactions containing each selected itemset again, without relying on the stored support value.

# For example, if 80 out of 1,000 transactions contain a selected itemset:
# \[
# \text{Direct support}=\frac{80}{1000}=0.08
# \]

# The direct support should match the Apriori support, allowing for display rounding.
# Expected output
# When you run the program, it will print:
# 1. The number of transactions and five example transactions.
# 2. \(L_1\), including frequent single items and their support.
# 3. \(L_2\), including frequent pairs and their support.
# 4. \(L_3\), including frequent triples and their support.
# 5. Three direct support verification results.
# The exact output depends on the CSV file.
# Important assumptions
# - Rating classes: The code uses Low for ratings below 3, Medium for ratings from 3 to below 4, and High for ratings of 4 or above. Change these cutoffs if your instructor specified different ones.
# - Review periods: Only records from 2006 through 2017 are included because those are the requested periods.
# - Cocoa bands: The code interprets the shorthand as two categories, below 70% and 70% or above.
# - Missing data: Rows missing any of the four required attributes are removed before creating transactions.
# Viva questions and answers
# 1. What is Apriori?

# Apriori is an algorithm for finding frequent itemsets in transactional data.

# 2. What is a transaction?

# A transaction is a collection of items associated with one observation. Here, one cocoa review is one transaction.

# 3. What is support?

# Support is the fraction of transactions containing a particular itemset.

# 4. What does minimum support 0.05 mean?

# The itemset must appear in at least 5% of transactions to be frequent.

# 5. What are \(L_1\), \(L_2\), and \(L_3\)?

# They represent frequent itemsets containing one, two, and three items, respectively.

# 6. What is the Apriori principle?

# Every non-empty subset of a frequent itemset must also be frequent.

# 7. Why are the remaining locations grouped as Other?

# To reduce the number of distinct location items and group less common locations together.

# 8. Why verify support manually?

# To confirm that the algorithm's support calculation matches a direct count of transactions.
# Conclusion
# The Cocoa dataset was converted into transactions containing four categorical items per review. The Apriori algorithm was implemented from scratch to identify frequent one-, two-, and three-itemsets using a minimum support threshold of 0.05. The support values of selected itemsets were then verified by directly counting the transactions in which they occurred.