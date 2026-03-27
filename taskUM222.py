import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/bmw_global_sales_2018_2025.csv")

#Analiza statystyczna z wybranymi columns
numeric_cols = ["Units_Sold", "Avg_Price_EUR", "Year"]
stats = data[numeric_cols].describe()
print("Opis statystyczny danych:\n", stats)

fig, ax = plt.subplots(1,3, figsize=(15,10))

# Wykres 1 – sprzedaż w kolejnych latach
subdata1 = data[["Year", "Units_Sold"]].dropna()
subdata1["Year"] = subdata1["Year"].astype(int)
ax1 = subdata1.plot(x="Year", y="Units_Sold", kind="line", color="blue", ax=ax[0], marker='o', title="Sprzedaż BMW w latach")

fig1 = ax1.get_figure()
fig1.savefig("salesInYears2.png")

# Wykres 2 – cena vs sprzedaż
subdata2 = data[["Units_Sold", "Avg_Price_EUR"]].dropna()
ax2 = subdata2.plot.scatter(x="Units_Sold", y="Avg_Price_EUR", color="red", ax=ax[1], title="Cena vs Sprzedaż")

fig2 = ax2.get_figure()
fig2.savefig("priceVsSell2.png")

# Wykres 3 – średnia cena modelu (groupby)
subdata3 = data[["Model", "Avg_Price_EUR"]].dropna()
model_price = subdata3.groupby("Model")["Avg_Price_EUR"].mean()
ax3 = model_price.plot(kind="bar", color="green", ax=ax[2], title="Średnia cena modelu")

fig3 = ax3.get_figure()
fig3.savefig("avg2.png")

plt.show()