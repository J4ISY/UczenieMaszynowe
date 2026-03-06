import pandas as pd

titanic = pd.read_csv("data/titanic.csv")

# print(titanic.head())

# Age in titanic members
age = titanic["Age"] # Intrested in age? Check this.
# print(age.head(7))

# print(type(titanic["Age"])) # As a single column is selected, the returned object is a pandas Series

# print(titanic["Age"].shape) # Shape

age_sex = titanic[["Age", "Sex"]] # Check Age and Sex

# print(age_sex.head(12))
# print(type(age_sex)) # panda.DataFrame
# print(age_sex.shape) # 891 rows, 2 columns

ageABV = titanic[titanic["Age"] > 35] # More than 35 age
ageABVB = titanic["Age"] > 35 # More than 35 age BOOL
# print(ageABV.head(13))
# print(ageABVB.head(13))

# print(ageABV.shape) # 217