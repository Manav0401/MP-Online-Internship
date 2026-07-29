import pandas as pd

# Load Dataset
df = pd.read_csv("data/cardekho_dataset.csv")

print("=" * 50)
print("Original Shape :", df.shape)

# Remove unwanted columns

df.drop(columns=["Unnamed: 0", "car_name"], inplace=True)

print("\nColumns after removing unwanted columns:\n")
print(df.columns)

# Remove duplicate rows

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates :", df.shape)

# Check Missing Values

print("\nMissing Values\n")

print(df.isnull().sum())

df.to_csv("data/cleaned_dataset.csv", index=False)

print("\nClean dataset saved successfully!")