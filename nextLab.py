import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit

data = pd.read_csv('data/bmw_global_sales_2018_2025.csv') # Import D A N Y C H

clean_data = data[['Year', 'Revenue_EUR', 'Units_Sold']] #Ustawienie kolumn (wybor)

clean_data = clean_data.dropna() #usunece pustych/ braków w kolumnach

x = clean_data[['Year', 'Revenue_EUR']] # Dane wejściowe
y = clean_data.Units_Sold # Ma przewidzeć sprzedać u.Solds

model = tree.DecisionTreeRegressor() # model drzewa decyzyjnego

model.fit(x, y) #ustawienie fitu

p1 = model.predict(x) #predykcja
# print(p1)


# Cala predykcja dla Data Frame
df = pd.DataFrame({
    'Year': [2025],
    'Revenue_EUR': [1000000],
})

p3 = model.predict(df)
# print(p3)

print('---------------------')
print('Mean Absolute Error:')
#Mean absolute error
mea = mean_absolute_error(y, p1)

print(p1.mean())
print(mea)
print((mea / p1.mean() * 100))

r2score = r2_score(y, p1)
print('r2')
print(r2score)

print('-------')
# train_test_split
print('Train Test Split')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model2 = tree.DecisionTreeRegressor()
model2.fit(x_train, y_train)

p2 = model2.predict(x_test)

mea = mean_absolute_error(y_test, p2)

print(p2.mean())
print(mea)
print((mea / p2.mean() *100))

r2score = r2_score(y_test, p2)

print(r2score)

# KFold
print('-------')
print('KFold')

kf = KFold(n_splits=5)

error_kfold = []

for train_index, test_index in kf.split(x):
    model3 = tree.DecisionTreeRegressor()
    model3.fit(x.iloc[train_index], y.iloc[train_index])

    y_pred = model3.predict(x.iloc[test_index])

    error_temp = mean_absolute_error(y.iloc[test_index], y_pred)
    error_kfold.append(error_temp)

print('Błędy KFold:', error_kfold)
print('Średni błąd KFold:', sum(error_kfold)/len(error_kfold))


# ShuffleSplit
print('-------')
print('ShuffleSplit')

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

error_shuffle = []

for train_index, test_index in ss.split(x):
    model4 = tree.DecisionTreeRegressor()
    model4.fit(x.iloc[train_index], y.iloc[train_index])

    y_pred = model4.predict(x.iloc[test_index])

    error_temp = mean_absolute_error(y.iloc[test_index], y_pred)
    error_shuffle.append(error_temp)

print('Błędy ShuffleSplit:', error_shuffle)
print('Średni błąd ShuffleSplit:', sum(error_shuffle)/len(error_shuffle))