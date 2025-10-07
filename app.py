from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Inisialisasi Flask
# ===============================
app = Flask(__name__)
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Dataset
# ===============================
data = {
    "Hours": [2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8],
    "Scores": [21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]
}
df_dataset = pd.DataFrame(data)

# ===============================
# Fungsi membuat semua grafik
# ===============================
def create_plots(df):
    plots = []

    # ===============================
    # Grafik lama (yang sudah ada)
    # ===============================
    # Scatter + regplot
    fig = plt.figure(figsize=(6,4))
    sns.regplot(x="Hours", y="Scores", data=df, ci=None,
                scatter_kws={"alpha":0.8}, line_kws={"color":"red"})
    plt.title("Scatter Plot + Regression Line")
    plt.xlabel("Hours Studied")
    plt.ylabel("Scores")
    plt.tight_layout()
    scatter_file = os.path.join(OUTPUT_DIR, "scatter_regplot.png")
    fig.savefig(scatter_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/scatter_regplot.png")

    # Histogram Scores
    fig = plt.figure(figsize=(6,4))
    sns.histplot(df["Scores"], bins=8, kde=True, color="lightgreen")
    plt.title("Histogram of Scores")
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_scores_file = os.path.join(OUTPUT_DIR, "hist_scores.png")
    fig.savefig(hist_scores_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/hist_scores.png")

    # Stripplot Scores
    fig = plt.figure(figsize=(6,4))
    sns.stripplot(y="Scores", data=df, size=8, color="green", jitter=0.1)
    plt.title("Distribution of Scores")
    plt.ylabel("Scores")
    plt.tight_layout()
    strip_scores_file = os.path.join(OUTPUT_DIR, "strip_scores.png")
    fig.savefig(strip_scores_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/strip_scores.png")

    # Heatmap korelasi
    fig = plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_file = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(heatmap_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/correlation_heatmap.png")

    # ===============================
    # Grafik baru dari kode kedua (hanya ditambahkan)
    # ===============================
    # Histogram Hours
    fig = plt.figure(figsize=(6,4))
    sns.histplot(df["Hours"], bins=8, kde=True, color="skyblue")
    plt.title("Histogram of Hours")
    plt.xlabel("Hours Studied")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_hours_file = os.path.join(OUTPUT_DIR, "hist_hours.png")
    fig.savefig(hist_hours_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/hist_hours.png")

    # Stripplot Hours
    fig = plt.figure(figsize=(6,4))
    sns.stripplot(y="Hours", data=df, size=8, color="blue", jitter=0.1)
    plt.title("Distribution of Hours")
    plt.ylabel("Hours Studied")
    plt.tight_layout()
    strip_hours_file = os.path.join(OUTPUT_DIR, "boxplot_hours.png")
    fig.savefig(strip_hours_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/boxplot_hours.png")

    # Scatter + LinearRegression (sklearn)
    from sklearn.linear_model import LinearRegression
    X = df[["Hours"]].values
    y = df["Scores"].values
    model = LinearRegression()
    model.fit(X, y)

    fig = plt.figure(figsize=(6,4))
    plt.scatter(X, y, alpha=0.8, color="purple")
    plt.plot(X, model.predict(X), color="red", linewidth=2)
    plt.title("Scatter Plot + Regression Line (LinearRegression)")
    plt.xlabel("Hours Studied")
    plt.ylabel("Scores")
    plt.tight_layout()
    linreg_file = os.path.join(OUTPUT_DIR, "scatter_linear_model.png")
    fig.savefig(linreg_file, dpi=150)
    plt.close(fig)
    plots.append("outputs/scatter_linear_model.png")

    return plots

# ===============================
# Route utama
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    plots = []
    matched_scores = []
    message = ""

    if request.method == "POST":
        hours_input = request.form.get("hours")
        try:
            hours_list = [float(x.strip()) for x in hours_input.split(",")]

            # Ambil scores sesuai dataset
            matched_scores = []
            for h in hours_list:
                matched = df_dataset[df_dataset["Hours"] == h]["Scores"].tolist()
                if matched:
                    matched_scores.append(matched[0])
                else:
                    matched_scores.append("Data tidak ada")  # jika hours tidak ada di dataset

            plots = create_plots(df_dataset)  # tampilkan semua grafik dari dataset asli

        except Exception as e:
            message = f"Error: {e}"

    return render_template("index.html",
                           plots=plots,
                           matched_scores=matched_scores,
                           message=message)

# ===============================
# Jalankan server
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
