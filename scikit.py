import pandas as pd
from sklearn import tree

data = pd.read_csv('data/melb_data.csv')

clean_data = data[['Price', 'Distance', 'Rooms', 'YearBuilt', 'Car', 'Lattitude', 'Longtitude']]

clean_data = clean_data.dropna()

x = clean_data[['Distance', 'Rooms', 'YearBuilt', 'Car', 'Lattitude', 'Longtitude']]
y = clean_data.Price

model = tree.DecisionTreeRegressor()
model.fit(x, y)
p1 = model.predict(x)

print(p1)

p2 = model.predict([[30.0,5,2000.0,3,-37.9999,144.9999]])
print(p2)

df = pd.DataFrame({
    'Distance': [30.0],
    'Rooms': [5],
    'YearBuilt': [2000],
    'Car': [3],
    'Lattitude': [-37.9999],
    'Longtitude': [144.9999],
})

p3 = model.predict(df)
print(p3)