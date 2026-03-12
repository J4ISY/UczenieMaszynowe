import pandas as pd

titanic = pd.read_csv("data/titanic.csv")

# print(titanic.head())

# print(titanic["Age"].mean()) # AVG Age of titanic passangers

# print(titanic[["Age", "Fare"]].median()) # median age & ticket fare price of Titanic passangers

# print(titanic[["Age", "Fare"]].describe()) # things like: count, mean, min, 25%, 50%, 75%, max

# print(titanic.agg( # Instead of the predefined statistics,
#     # specific combinations of aggregating statistics for given columns can be defined using the DataFrame.agg()
#     {
#         "Age": ["min", "max", "median", "skew"],
#         "Fare": ["min", "max", "median", "mean"],
#     }
# ))

# print(titanic[["Sex", "Age"]].groupby("Sex").mean()) # AVG age for male vs female of titanic passangers

# print(titanic.groupby("Sex").mean(numeric_only=True))

# print(titanic.groupby("Sex")["Age"].mean())

# print(titanic.groupby(["Sex", "Pclass"])["Fare"].mean()) # mean ticket fare price for each sex and cabin class combination

# print(titanic["Pclass"].value_counts()) # number of passangers in each of cabin classes

print(titanic.groupby("Pclass")["Pclass"].count())