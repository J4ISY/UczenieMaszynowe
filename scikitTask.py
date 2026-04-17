import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/bmw_global_sales_2018_2025.csv')

clean_data = data[['Year', 'Units_Sold', 'Avg_Price_EUR', 'Revenue_EUR', 'GDP_Growth']]

clean_data = clean_data.dropna()

x = clean_data[['Year', 'Avg_Price_EUR', 'GDP_Growth', 'Revenue_EUR']]
y = clean_data.Units_Sold

model = tree.DecisionTreeRegressor()

model.fit(x, y)

p1 = model.predict(x)

mea = mean_absolute_error(y, p1)

print(p1.mean())
print(mea)
print((mea / p1.mean() * 100))

r2score = r2_score(y, p1)

print(r2score)

print('---------------------------------------------------------------------------')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model2 = tree.DecisionTreeRegressor()
model2.fit(x_train, y_train)

p2 = model2.predict(x_test)

mea = mean_absolute_error(y_test, p2)

print(p2.mean())
print(mea)
print((mea / p2.mean() * 100))

r2score = r2_score(y_test, p2)

print(r2score)