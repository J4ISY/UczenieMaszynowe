import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/melb_data.csv")

subdata1 = data[["Rooms", "Price"]]
subdata2 = data[["YearBuilt", "Price"]]

print(subdata1.head())

subdata2 = subdata2.dropna()

subdata2["YearBuilt"] = subdata2["YearBuilt"].astype(int)

ax1 = subdata1.plot(x="Rooms", y="Price", kind="scatter")
plt.show()

ax2 = subdata2.plot(x="YearBuilt", y="Price", kind="scatter")
plt.show()

subdata2_grouped = subdata2.groupby("YearBuilt")["Price"].mean()

ax3 = subdata2_grouped.plot(x="YearBuilt", y="Price", kind="bar")
plt.show()

fig = ax3.get_figure()
fig.savefig("fig1.png")
print("")