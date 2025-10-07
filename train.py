# ==================================================
# PENGEMBANGAN MODEL REGRESI LINEAR
# ==================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # ✅ Tambahkan ini untuk menyimpan model

# ==================================================
# 1️⃣ Muat Dataset
# ==================================================
data = {
    "Hours": [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
    "Scores": [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]
}
df = pd.DataFrame(data)

# ==================================================
# 2️⃣ Bagi Data Menjadi Training dan Testing
# ==================================================
X = df[["Hours"]]  # variabel independen
y = df["Scores"]   # variabel dependen

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# ==================================================
# 3️⃣ Latih Model LinearRegression
# ==================================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==================================================
# 4️⃣ ✅ Simpan Model Menggunakan Joblib (Model Persistence)
# ==================================================
joblib.dump(model, 'model_regresi_linear.joblib')
print("\nModel berhasil disimpan sebagai 'model_regresi_linear.joblib'")

# ==================================================
# 5️⃣ Prediksi pada Data Testing
# ==================================================
y_pred = model.predict(X_test)

# ==================================================
# 6️⃣ Evaluasi Model
# ==================================================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Evaluasi Model ===")
print(f"R-squared (R²): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# ==================================================
# 7️⃣ Interpretasi Koefisien dan Intersep
# ==================================================
print("\n=== Parameter Model ===")
print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Koefisien (b1): {model.coef_[0]:.4f}")

# Penjelasan Kontekstual
print("\n=== Interpretasi ===")
print(f"Artinya: Jika jam belajar (Hours) meningkat 1 jam, maka nilai (Scores) "
      f"diperkirakan naik sebesar {model.coef_[0]:.2f} poin.")
print(f"Sedangkan ketika jam belajar = 0, prediksi nilai awal (intersep) adalah sekitar {model.intercept_:.2f} poin.")

# ==================================================
# 8️⃣ Visualisasi Hasil
# ==================================================
plt.figure(figsize=(7,5))
sns.scatterplot(x="Hours", y="Scores", data=df, color="purple", label="Data Asli")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Garis Prediksi (Test Data)")
plt.title("Linear Regression - Hubungan Jam Belajar vs Nilai")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ==================================================
# 9️⃣ Tabel Perbandingan Prediksi vs Aktual
# ==================================================
comparison = pd.DataFrame({
    "Hours": X_test["Hours"].values,
    "Actual Scores": y_test.values,
    "Predicted Scores": y_pred
})
print("\n=== Perbandingan Nilai Aktual vs Prediksi ===")
print(comparison.round(2))
