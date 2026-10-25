
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# If mlxtend is not installed, uncomment the next line and run it:
# %pip install mlxtend

# 1. Load the GHI dataset
df = pd.read_csv("GHI_Report (1).csv")
df.columns = df.columns.str.strip()

# 2. Select the columns used in the practical
columns = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=columns).copy()
df = df.reset_index(drop=True)

print("Number of countries:", len(df))

# 3. Convert each numerical attribute into Low, Medium, or High
# Equal-frequency binning creates three groups of approximately equal size.
for col in columns:
    categories = pd.qcut(
        df[col].rank(method="first"),
        q=3,
        labels=["Low", "Medium", "High"]
    )

    if col == "H_Score":
        df["Score_Level"] = "Score-" + categories.astype(str)
    else:
        df[col + "_Level"] = col + "-" + categories.astype(str)

# 4. Create one transaction per country
item_columns = [
    "Economy_Level",
    "Fam_Level",
    "Health_Level",
    "Freedom_Level",
    "Score_Level"
]

transactions = df[item_columns].values.tolist()

print("\nExample transactions:")
for transaction in transactions[:5]:
    print(transaction)

# 5. Convert transactions into a one-hot encoded table
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

# 6. Generate frequent itemsets using minimum support = 0.10
min_support = 0.10

frequent_itemsets = apriori(
    transaction_df,
    min_support=min_support,
    use_colnames=True
)

print("\nFrequent itemsets:")
if frequent_itemsets.empty:
    print("No frequent itemsets found at the selected minimum support.")
else:
    frequent_itemsets["Itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda x: ", ".join(sorted(x))
    )
    print(
        frequent_itemsets[["Itemsets", "support"]]
        .sort_values("support", ascending=False)
        .round(3)
        .to_string(index=False)
    )

# 7. Generate association rules with minimum confidence = 0.60
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets.drop(columns=["Itemsets"], errors="ignore"),
        metric="confidence",
        min_threshold=0.60
    )

    # Keep only rules whose consequent is exactly Score-High
    rules = rules[
        rules["consequents"].apply(
            lambda items: items == frozenset(["Score-High"])
        )
    ].copy()

    if rules.empty:
        print(
            "\nNo rules with consequent Score-High satisfy both "
            "minimum support 0.10 and minimum confidence 0.60."
        )
    else:
        # 8. Prepare readable rule names and rank by lift
        rules["Antecedent"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules["Consequent"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        rules = rules.sort_values("lift", ascending=False)

        print("\nAssociation rules with consequent Score-High")
        print("Minimum support =", min_support)
        print("Minimum confidence = 0.60\n")

        print(
            rules[
                [
                    "Antecedent",
                    "Consequent",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].round(3).to_string(index=False)
        )

        # 9. Interpret the top-ranked rule
        best_rule = rules.iloc[0]

        print("\nInterpretation of the highest-lift rule:")
        print("If:", best_rule["Antecedent"])
        print("Then:", best_rule["Consequent"])
        print("Support:", round(best_rule["support"], 3))
        print("Confidence:", round(best_rule["confidence"], 3))
        print("Lift:", round(best_rule["lift"], 3))

        if best_rule["lift"] > 1:
            print(
                "Interpretation: The antecedent is positively associated "
                "with Score-High. Countries matching the antecedent are "
                "more likely to have a high score than the overall dataset."
            )
        elif best_rule["lift"] < 1:
            print(
                "Interpretation: The antecedent is negatively associated "
                "with Score-High compared with the overall dataset."
            )
        else:
            print(
                "Interpretation: The antecedent does not show an association "
                "with Score-High beyond the overall baseline."
            )


# Aim: To generate association rules from the Global Happiness Index (GHI) dataset using minimum support = 0.10 and minimum confidence = 0.60, then select rules whose consequent is Score-High, rank them by lift, and interpret the strongest rules.

# We will use the practical dataset filename GHI_Report (1).csv and the columns Economy, Fam, Health, Freedom, and H_Score.
# Concepts and formulas
# - Transaction: One country's record, converted into categorical items such as Economy-High or Health-Medium.
# - Support: The proportion of countries containing all items in a rule.
# \[
# \operatorname{Support}(A\rightarrow B)=
# \frac{\operatorname{Count}(A\cup B)}{N}
# \]

# - Confidence: The probability of finding \(B\) among countries that contain \(A\).
# \[
# \operatorname{Confidence}(A\rightarrow B)=
# \frac{\operatorname{Support}(A\cup B)}
# {\operatorname{Support}(A)}
# \]

# - Lift: Measures how strongly \(A\) and \(B\) occur together compared with what would be expected if they were independent.
# \[
# \operatorname{Lift}(A\rightarrow B)=
# \frac{\operatorname{Confidence}(A\rightarrow B)}
# {\operatorname{Support}(B)}
# \]

# Interpretation of lift:
# - Lift greater than 1: positive association.
# - Lift equal to 1: no association beyond what is expected by chance.
# - Lift less than 1: negative association.
# Explanation step by step
# 1. Data preprocessing
# The code loads GHI_Report (1).csv, removes spaces from column names, converts the five attributes to numeric values, and removes rows that have missing values in those columns.

# The columns are:
# Column	Meaning
# Economy	Economy-related score
# Fam	Family/social support score
# Health	Health-related score
# Freedom	Freedom-related score
# H_Score	Happiness score

# 2. Discretization
# Association rule mining works with items or categories rather than continuous numerical values. So each attribute is divided into three approximately equal-sized groups using equal-frequency binning.

# For example:
# - Economy-Low
# - Economy-Medium
# - Economy-High
# - Health-Low
# - Health-Medium
# - Health-High
# - Score-Low
# - Score-Medium
# - Score-High
# The categories are determined from the dataset's distribution. Therefore, Score-High here means the upper third of the H_Score values, not a fixed happiness cutoff such as 6.5.
# 3. Transactions
# Each country becomes one transaction containing five items. For example, a country might have:

# Economy-High, Fam-Medium, Health-High, Freedom-Medium, Score-High

# These items describe the characteristics of that country.
# 4. Frequent itemsets
# The Apriori algorithm finds combinations of items that occur often enough. The minimum support is set to 0.10, meaning an itemset must appear in at least 10% of the countries.
# \[
# \text{Support}(X)=\frac{\text{Countries containing }X}{\text{Total countries}}
# \]

# For example, if 30 out of 200 countries contain both Economy-High and Score-High, their support is:
# \[
# \frac{30}{200}=0.15
# \]

# That itemset satisfies the minimum support threshold.
# 5. Association rules
# An association rule has the form:
# \[
# A \rightarrow B
# \]

# For this practical, the consequent \(B\) must be exactly Score-High. A possible example is:

# Economy-High, Health-High → Score-High

# This is an illustrative example, not a claim about your actual results. The code discovers the rules from your CSV file.

# The minimum confidence is 0.60, so at least 60% of the countries matching the antecedent must also have Score-High.
# \[
# \text{Confidence}(A\rightarrow B)=\frac{P(A\cup B)}{P(A)}
# \]

# 6. Ranking rules by lift
# The qualifying rules are sorted from highest to lowest lift.
# \[
# \text{Lift}(A\rightarrow B)=
# \frac{\text{Confidence}(A\rightarrow B)}
# {\text{Support}(B)}
# \]

# A higher lift indicates a stronger positive association with Score-High relative to its overall frequency. Lift does not establish causation.
# Expected output
# Your notebook will display:
# 1. The number of countries included.
# 2. A few example transactions.
# 3. Frequent itemsets and their support values.
# 4. Rules with Score-High as the consequent.
# 5. Support, confidence, and lift for each qualifying rule.
# 6. An interpretation of the highest-lift rule.
# If no rules appear, that means no rules meet both specified thresholds for this dataset and discretization; do not interpret that as proof that no relationships exist.
# Viva questions and answers
# 1. What is association rule mining?

# It is a data-mining technique used to discover relationships and co-occurrence patterns among items in a dataset.

# 2. What is the Apriori algorithm?

# Apriori identifies frequent itemsets and uses the principle that every subset of a frequent itemset must also be frequent.

# 3. What is minimum support?

# It is the minimum fraction of transactions in which an itemset must occur. Here it is 0.10, or 10%.

# 4. What is minimum confidence?

# It is the minimum conditional probability required for a rule. Here it is 0.60, or 60%.

# 5. Why use lift?

# Lift compares the rule's confidence with the overall support of its consequent. Lift above 1 indicates a positive association.

# 6. Why discretize numerical values?

# To convert numerical attributes into categorical items that can be used in transaction-based association rule mining.

# Conclusion: This experiment applies Apriori-based association rule mining to the GHI dataset, retains rules predicting Score-High that meet the given support and confidence thresholds, and ranks them by lift to identify the strongest associations.