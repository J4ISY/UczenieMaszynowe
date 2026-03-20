import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/melb_data.csv")

subdata1 = data[["Rooms", "Price", "Bathroom"]]
subdata2 = data[["YearBuilt", "Price"]]


# print(subdata1.head())

subdata2 = subdata2.dropna()

subdata2["YearBuilt"] = subdata2["YearBuilt"].astype(int)

ax1 = subdata1.plot(x="Rooms", y="Price", kind="scatter")
# plt.show()

ax2 = subdata2.plot(x="YearBuilt", y="Price", kind="scatter")
# plt.show()

subdata2_grouped = subdata2.groupby("YearBuilt")["Price"].mean()

ax3 = subdata2_grouped.plot(x="YearBuilt", y="Price", kind="bar")
# plt.show()

# fig = ax3.get_figure()
# fig.savefig("fig1.png")

new_gr = subdata1.loc[subdata1["Rooms"] > 2]

only1bath = data.loc[data.Bathroom == 1, ["Price", "YearBuilt"]]

only1bath0car = data.loc[(data.Bathroom == 1) & (data.Car == 0), ['Price', 'YearBuilt']]

subdata1.loc[subdata1['Bathroom'] > 2, 'Size'] = 'big'
subdata1.loc[subdata1['Bathroom'] <= 2, 'Size'] = 'small'
# print(new_gr.head(10))

values = data['Suburb'].unique()
# stowrzyć nowy data frame, bedzie mial nazyw ich, a w drugiej ilosc ich wystąpień

subdata3 = data[["Suburb"]]

values = data['Suburb'].unique()
wynik = pd.DataFrame(columns=['Suburb', 'Count'])

for i, suburb in enumerate(values):
    ilosc = data.loc[data.Suburb == suburb].shape[0]
    wynik.loc[i, 'Suburb'] = suburb
    wynik.loc[i, 'Count'] = ilosc

print(wynik.head(10))