import pandas as pd
from sklearn import tree

data = pd.read_csv('data/bmw_global_sales_2018_2025.csv') # Import D A N Y C H

clean_data = data[['Year', 'Revenue_EUR', 'Units_Sold']] #Ustawienie kolumn (wybor)

clean_data = clean_data.dropna() #usunece pustych/ braków w kolumnach

x = clean_data[['Year', 'Revenue_EUR']] # Dane wejściowe
y = clean_data.Units_Sold # Ma przewidzeć sprzedać u.Solds

model = tree.DecisionTreeRegressor() # model drzewa decyzyjnego czy cos takiego bylo

model.fit(x, y) #ustawienie fitu

p1 = model.predict(x) #predykcja
print(p1)

p2 = model.predict([[2025, 1000000]]) #predykcja 2
print(p2)

# Cala predykcja dla Data Frame
df = pd.DataFrame({
    'Year': [2025],
    'Revenue_EUR': [1000000],
})

p3 = model.predict(df)
print(p3)