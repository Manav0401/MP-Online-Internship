import pandas as pd

df = pd.read_csv("data/cardekho_dataset.csv")
# Unique Values in Each Column

print("\n Unique Values \n")

for column in df.columns:
    print(f"{column} : {df[column].nunique()} unique values")

# Brand Information

print("\n Brands")

print("Total Brands :", df["brand"].nunique())

print("\nList of Brands:\n")
print(sorted(df["brand"].unique()))

# Model Information

print("\n Models")

print("Total Models :", df["model"].nunique())

# Cars per Brand

print("\n Cars per Brand \n")

print(df["brand"].value_counts())

# Fuel Types

print("\n Fuel Types \n")

print(df["fuel_type"].value_counts())

# Transmission Types

print("\n Transmission \n")

print(df["transmission_type"].value_counts())


# Seller Types

print("\n========== Seller Types\n")

print(df["seller_type"].value_counts())