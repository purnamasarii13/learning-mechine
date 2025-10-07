import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# ===============================
# Dataset
# ===============================
data = {
    "Hours": [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
    "Scores": [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]
}
df = pd.DataFrame(data)

# ===============================
# Folder output
# ===============================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_and_show(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()

# ===============================
# 1️⃣ Eksplorasi Awal Dataset
# ===============================
print("=== INFORMASI DATASET ===")
print(df.info())
print("\n=== 5 DATA TERATAS ===")
print(df.head())
print("\n=== DESKRIPSI STATISTIK ===")
print(df.describe())

# ===============================
# 2️⃣ Cek Missing Values
# ===============================
print("\n=== CEK NILAI KOSONG ===")
print(df.isnull().sum())

# Jika ada nilai kosong, kita bisa isi dengan mean (contoh)
df = df.fillna(df.mean())

# ===============================
# 3️⃣ Deteksi Outlier (Z-Score)
# ===============================
z_scores = np.abs((df - df.mean()) / df.std())
outliers = df[(z_scores > 3).any(axis=1)]
print("\n=== DATA OUTLIERS (Z-Score > 3) ===")
print(outliers if not outliers.empty else "Tidak ada outlier signifikan.")

# ===============================
# 4️⃣ Scatter plot (validasi tren linear)
# ===============================
fig = plt.figure(figsize=(6,4))
sns.regplot(x="Hours", y="Scores", data=df, ci=None, scatter_kws={"alpha":0.8}, line_kws={"color":"red"})
plt.title("Scatter Plot + Regression Line (Validasi Tren Linear)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.tight_layout()
save_and_show(fig, "scatter_regplot.png")

# Korelasi Pearson (untuk validasi tren linear)
corr = df["Hours"].corr(df["Scores"])
print(f"\nKoefisien Korelasi Pearson antara Hours dan Scores: {corr:.3f}")

# ===============================
# 5️⃣ Histogram (Distribusi Data)
# ===============================
fig = plt.figure(figsize=(6,4))
sns.histplot(df["Hours"], bins=8, kde=True, color="skyblue")
plt.title("Histogram of Hours")
plt.xlabel("Hours Studied")
plt.ylabel("Count")
plt.tight_layout()
save_and_show(fig, "hist_hours.png")

fig = plt.figure(figsize=(6,4))
sns.histplot(df["Scores"], bins=8, kde=True, color="lightgreen")
plt.title("Histogram of Scores")
plt.xlabel("Scores")
plt.ylabel("Count")
plt.tight_layout()
save_and_show(fig, "hist_scores.png")

# ===============================
# 6️⃣ Stripplot (Sebagai alternatif boxplot)
# ===============================
fig = plt.figure(figsize=(6,4))
sns.stripplot(y="Hours", data=df, size=8, color="blue", jitter=0.1)
plt.title("Distribution of Hours")
plt.ylabel("Hours Studied")
plt.tight_layout()
save_and_show(fig, "strip_hours.png")

fig = plt.figure(figsize=(6,4))
sns.stripplot(y="Scores", data=df, size=8, color="green", jitter=0.1)
plt.title("Distribution of Scores")
plt.ylabel("Scores")
plt.tight_layout()
save_and_show(fig, "strip_scores.png")

# ===============================
# 7️⃣ Heatmap Korelasi
# ===============================
fig = plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
save_and_show(fig, "correlation_heatmap.png")

# ===============================
# 8️⃣ Scatter Plot + LinearRegression (Model)
# ===============================
X = df[["Hours"]].values
y = df["Scores"].values
model = LinearRegression()
model.fit(X, y)

fig = plt.figure(figsize=(6,4))
plt.scatter(X, y, alpha=0.8, color="purple")
plt.plot(X, model.predict(X), color="red", linewidth=2)
plt.title("Linear Regression Fit (Sklearn)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.tight_layout()
save_and_show(fig, "linear_regression_fit.png")
