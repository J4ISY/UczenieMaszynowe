import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych
data = pd.read_csv("ram_pricing_intelligence_2026.csv")

# Wybrane kolumny
subdata = data[["capacity_gb", "price"]]

subdata2 = data[["bus_speed_mhz", "price"]]

subdata3 = data[["sold", "price"]]

# Usunięcie brakujących danych
subdata = subdata.dropna()
subdata2 = subdata2.dropna()
subdata3 = subdata3.dropna()

# Grupowanie danych
grouped = data.groupby("capacity_gb")["price"].mean()

# Wykres 1
ax1 = subdata.plot(
    x="capacity_gb",
    y="price",
    kind="scatter",
    title="RAM Capacity vs Price"
)
plt.savefig('images/CapacityGBvsPrice.png')
# Wykres 2
ax2 = subdata2.plot(
    x="bus_speed_mhz",
    y="price",
    kind="scatter",
    title="Bus Speed vs Price"
)
plt.savefig('images/BusSpeedMHZvsPrice.png')
# Wykres 3
ax3 = subdata3.plot(
    x="sold",
    y="price",
    kind="scatter",
    title="Units Sold vs Price"
)
plt.savefig('images/SoldvsPrice.png')
# Wykres liniowy
plt.figure()

ax4 = grouped.plot(
    kind="line",
    title="Average Price by RAM Capacity"
)
ax4.set_xlabel("Capacity (GB)")
ax4.set_ylabel("Average Price")
plt.savefig('images/AVGPricebyRamCapacity.png')
# Wyświetlenie wszystkich wykresów
plt.show()