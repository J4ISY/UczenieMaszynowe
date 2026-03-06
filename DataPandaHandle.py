import pandas as pd

df = pd.DataFrame(
     {
         "Name": [
             "Braund, Mr. Owen Harris",
             "Allen, Mr. William Henry",
             "Bonnell, Miss Elizabeth",
             "Mirchwasser, Mr. Claudette wassa von funf"
         ],
         "Age": [22, 35, 58, 67],
         "Sex": ["male", "male", "female", "male"],
     }
 )

dx = pd.DataFrame(
    {
        "Age": [67, 69, 27, 33],
        "Country": ["Poland", "United Stated", "United Kingdom", "France"],
    }
)

ages = pd.Series([22, 35, 58], name="Age")

# print(df)
# print(dx)
# print("Age max is: ", dx["Age"].max())
# print(dx.describe()) # Basic statistics
print(ages)