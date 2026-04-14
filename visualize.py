import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("WineQT.csv")

# Drop Id column if exists
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)


# -----------------------------
# 1. Countplot (Quality distribution)
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="quality", data=df)
plt.title("Wine Quality Distribution")
plt.show()


# -----------------------------
# 2. Scatter Plot (Alcohol vs Quality)
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(df["alcohol"], df["quality"])
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Alcohol vs Quality")
plt.show()


# -----------------------------
# 3. Histogram (Alcohol distribution)
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["alcohol"], bins=30)
plt.title("Alcohol Distribution")
plt.show()


# -----------------------------
# 4. Boxplot (Volatile acidity vs Quality)
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="quality", y="volatile acidity", data=df)
plt.title("Volatile Acidity vs Quality")
plt.show()


# -----------------------------
# 5. Violin Plot (Sulphates vs Quality)
# -----------------------------
plt.figure(figsize=(6,4))
sns.violinplot(x="quality", y="sulphates", data=df)
plt.title("Sulphates vs Quality")
plt.show()


# -----------------------------
# 6. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()