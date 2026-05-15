import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

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

print('======================================')
print('PORÓWNANIE MODELI')
print('======================================')

# -----------------------------------
# Przygotowanie train/test
# -----------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# 1. DecisionTree vs RandomForest
# ===================================

print('-------')
print('DecisionTree vs RandomForest')

dt_model = tree.DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)

dt_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)

dt_pred = dt_model.predict(x_test)
rf_pred = rf_model.predict(x_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

dt_r2 = r2_score(y_test, dt_pred)
rf_r2 = r2_score(y_test, rf_pred)

print('DecisionTree MAE:', dt_mae)
print('DecisionTree R2:', dt_r2)

print('RandomForest MAE:', rf_mae)
print('RandomForest R2:', rf_r2)

# Wykres MAE
plt.figure(figsize=(8,5))
plt.bar(['DecisionTree', 'RandomForest'], [dt_mae, rf_mae])
plt.title('Porównanie MAE')
plt.ylabel('MAE')
plt.savefig('./images/DecisionTreeVsRandomForestMAE.png')
plt.show()

# Wykres R2
plt.figure(figsize=(8,5))
plt.bar(['DecisionTree', 'RandomForest'], [dt_r2, rf_r2])
plt.title('Porównanie R2 Score')
plt.ylabel('R2')
plt.savefig('./images/DecisionTreeVsRandomForesR2.png')
plt.show()

# ===================================
# 2. max_depth
# ===================================

print('-------')
print('max_depth')

depths = [1, 2, 3, 4, 5, 10, 15]

dt_mae_depth = []
rf_mae_depth = []

dt_r2_depth = []
rf_r2_depth = []

for depth in depths:

    # Decision Tree
    dt = tree.DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    dt.fit(x_train, y_train)

    dt_pred = dt.predict(x_test)

    dt_mae_depth.append(
        mean_absolute_error(y_test, dt_pred)
    )

    dt_r2_depth.append(
        r2_score(y_test, dt_pred)
    )

    # Random Forest
    rf = RandomForestRegressor(
        max_depth=depth,
        random_state=42
    )

    rf.fit(x_train, y_train)

    rf_pred = rf.predict(x_test)

    rf_mae_depth.append(
        mean_absolute_error(y_test, rf_pred)
    )

    rf_r2_depth.append(
        r2_score(y_test, rf_pred)
    )

    print('Depth:', depth)
    print('DT MAE:', dt_mae_depth[-1])
    print('DT R2:', dt_r2_depth[-1])

    print('RF MAE:', rf_mae_depth[-1])
    print('RF R2:', rf_r2_depth[-1])

    print('----------------')

# Wykres MAE dla max_depth
plt.figure(figsize=(10,5))
plt.plot(depths, dt_mae_depth, marker='o', label='DecisionTree')
plt.plot(depths, rf_mae_depth, marker='o', label='RandomForest')

plt.title('MAE dla różnych max_depth')
plt.xlabel('max_depth')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.savefig('./images/MaxDepthMAE.png')
plt.show()

# Wykres R2 dla max_depth
plt.figure(figsize=(10,5))
plt.plot(depths, dt_r2_depth, marker='o', label='DecisionTree')
plt.plot(depths, rf_r2_depth, marker='o', label='RandomForest')

plt.title('R2 Score dla różnych max_depth')
plt.xlabel('max_depth')
plt.ylabel('R2 Score')
plt.legend()
plt.grid()
plt.savefig('./images/MaxDepthR2.png')
plt.show()

# ===================================
# 3. max_leaf_nodes
# ===================================

print('-------')
print('max_leaf_nodes')

leafs = [2, 5, 10, 20, 50, 100]

dt_mae_leaf = []
rf_mae_leaf = []

dt_r2_leaf = []
rf_r2_leaf = []

for leaf in leafs:

    # Decision Tree
    dt = tree.DecisionTreeRegressor(
        max_leaf_nodes=leaf,
        random_state=42
    )

    dt.fit(x_train, y_train)

    dt_pred = dt.predict(x_test)

    dt_mae_leaf.append(
        mean_absolute_error(y_test, dt_pred)
    )

    dt_r2_leaf.append(
        r2_score(y_test, dt_pred)
    )

    # Random Forest
    rf = RandomForestRegressor(
        max_leaf_nodes=leaf,
        random_state=42
    )

    rf.fit(x_train, y_train)

    rf_pred = rf.predict(x_test)

    rf_mae_leaf.append(
        mean_absolute_error(y_test, rf_pred)
    )

    rf_r2_leaf.append(
        r2_score(y_test, rf_pred)
    )

    print('Leaf nodes:', leaf)

    print('DT MAE:', dt_mae_leaf[-1])
    print('DT R2:', dt_r2_leaf[-1])

    print('RF MAE:', rf_mae_leaf[-1])
    print('RF R2:', rf_r2_leaf[-1])

    print('----------------')

# Wykres MAE dla leaf nodes
plt.figure(figsize=(10,5))
plt.plot(leafs, dt_mae_leaf, marker='o', label='DecisionTree')
plt.plot(leafs, rf_mae_leaf, marker='o', label='RandomForest')

plt.title('MAE dla różnych max_leaf_nodes')
plt.xlabel('max_leaf_nodes')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.savefig('./images/LeafNodesMAE.png')
plt.show()

# Wykres R2 dla leaf nodes
plt.figure(figsize=(10,5))
plt.plot(leafs, dt_r2_leaf, marker='o', label='DecisionTree')
plt.plot(leafs, rf_r2_leaf, marker='o', label='RandomForest')

plt.title('R2 Score dla różnych max_leaf_nodes')
plt.xlabel('max_leaf_nodes')
plt.ylabel('R2 Score')
plt.legend()
plt.grid()
plt.savefig('./images/LeafNodesR2.png')
plt.show()

print(clean_data.describe())