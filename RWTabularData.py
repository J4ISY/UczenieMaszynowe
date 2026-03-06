import pandas as pd

titanic = pd.read_csv("data/titanic.csv")
titanicXLSX = pd.read_excel("data/titanic.xlsx", sheet_name="passengers")

# print(titanic) # First 5 and Last 5 rows
# print(titanic.head(50)) # 50 Heads
# print(titanic.dtypes) # Data Types
# print(titanic.info()) # Technical info
# titanic.to_excel("data/titanic.xlsx", sheet_name="passengers", index=False) # File XLSX EXCEL
# print(titanicXLSX.head())