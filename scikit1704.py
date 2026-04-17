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