import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('GOOGL_2004-08-01_2024-12-18.csv')

# jeśli nazwy kolumn mają spacje, dobrze je uprościć
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# wybór tylko kolumn numerycznych
numeric_data = data.select_dtypes(include=['number'])

# macierz korelacji
corr = numeric_data.corr()

# heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation')

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title('Heatmap korelacji cech')
plt.tight_layout()
plt.savefig('images/HeatMap.png')
plt.show()