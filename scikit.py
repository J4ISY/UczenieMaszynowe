import pandas as pd

data = pd.read_csv('data/melb_data.csv')

clean_data = data[['Price', 'Distance', 'Rooms', 'Car', 'Lattitude', 'Longitude']]

clean_data = clean_data.dropna()

x = clean_data[['Distance', 'Rooms', 'YearBuilt', 'Lattitude', 'Longitude']]
y = clean_data.Price

