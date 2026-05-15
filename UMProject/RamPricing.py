import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =========================
# Wczytanie danych
# =========================

df = pd.read_csv("ram_pricing_intelligence_2026.csv")

# =========================
# Wybrane cechy
# =========================

X = df[[
    "ram_generation",
    "brand",
    "capacity_gb",
    "bus_speed_mhz",
    "unit_type",
    "condition",
    "wasPrice",
    "available",
    "sold",
    "itemLocation",
    "is_bulk_server"
]]

# Target
y = df["price"]

# =========================
# Kolumny
# =========================

categorical_features = [
    "ram_generation",
    "brand",
    "unit_type",
    "condition",
    "itemLocation"
]

numeric_features = [
    "capacity_gb",
    "bus_speed_mhz",
    "wasPrice",
    "available",
    "sold",
    "is_bulk_server"
]

# =========================
# Preprocessing
# =========================

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features)
    ]
)

# =========================
# Model
# =========================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# =========================
# Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Trenowanie
# =========================

model.fit(X_train, y_train)

# =========================
# Predykcja
# =========================

predictions = model.predict(X_test)

# =========================
# Ocena modelu
# =========================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", round(mae, 2))
print("R2:", round(r2, 4))



plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted RAM Prices")
plt.show()