import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/bmw_global_sales_2018_2025.csv")

# Wykres 1 – sprzedaż w kolejnych latach

subdata1 = data[["Year", "Units_Sold"]]
subdata1 = subdata1.dropna()  # usunięcie pustych wartości
subdata1["Year"] = subdata1["Year"].astype(int)  # poprawa typu danych

ax1 = subdata1.plot(x="Year", y="Units_Sold",kind="line")
plt.show()

fig1 = ax1.get_figure()
fig1.savefig("salesInYears.png")

# Wykres 2 – cena vs sprzedaż

subdata2 = data[["Units_Sold", "Avg_Price_EUR"]]
subdata2 = subdata2.dropna()

ax2 = subdata2.plot(x="Units_Sold", y="Avg_Price_EUR",kind="scatter")
plt.show()

fig2 = ax2.get_figure()
fig2.savefig("priceVsSales.png")

# Wykres 3 – średnia cena modelu (groupby)

subdata3 = data[["Model", "Avg_Price_EUR"]]
subdata3 = subdata3.dropna()

model_price = subdata3.groupby("Model")["Avg_Price_EUR"].mean()

ax3 = model_price.plot(kind="bar")
plt.show()

fig3 = ax3.get_figure()
fig3.savefig("AVGPriceByModel.png")