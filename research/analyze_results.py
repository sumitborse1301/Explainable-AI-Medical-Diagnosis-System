import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
CSV_PATH = "results/chest_eval_results.csv"
OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD RESULTS
# =========================
print("[INFO] Loading results CSV...")
df = pd.read_csv(CSV_PATH)

print("[INFO] Total samples:", len(df))
print(df.head())

# =========================
# CLEAN DATA
# =========================
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df = df.dropna(subset=["prediction", "confidence"])

# =========================
# 1. PATHOLOGY FREQUENCY
# =========================
freq = df["prediction"].value_counts()

freq_path = os.path.join(OUT_DIR, "pathology_frequency.csv")
freq.to_csv(freq_path)

print("\n[RESULT] Top predicted pathologies:")
print(freq.head(10))

# Plot top 10
plt.figure(figsize=(10,5))
freq.head(10).plot(kind="bar")
plt.title("Top 10 Predicted Chest Pathologies")
plt.ylabel("Number of Images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top10_pathology_distribution.png"))
plt.close()

# =========================
# 2. CONFIDENCE STATISTICS
# =========================
mean_conf = df.groupby("prediction")["confidence"].mean().sort_values(ascending=False)
std_conf = df.groupby("prediction")["confidence"].std()

conf_stats = pd.concat([mean_conf, std_conf], axis=1)
conf_stats.columns = ["mean_confidence", "std_confidence"]

conf_path = os.path.join(OUT_DIR, "confidence_stats.csv")
conf_stats.to_csv(conf_path)

print("\n[RESULT] Mean confidence per pathology (top 10):")
print(conf_stats.head(10))

# =========================
# 3. CONFIDENCE DISTRIBUTION
# =========================
plt.figure(figsize=(8,5))
df["confidence"].hist(bins=30)
plt.title("Overall Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confidence_distribution.png"))
plt.close()

# =========================
# 4. LOW CONFIDENCE CASES (FAILURE ZONE)
# =========================
low_conf = df[df["confidence"] < 0.55]

low_conf_path = os.path.join(OUT_DIR, "low_confidence_cases.csv")
low_conf.to_csv(low_conf_path, index=False)

print("\n[RESULT] Low-confidence cases:", len(low_conf))
print("Saved:", low_conf_path)

# =========================
# DONE
# =========================
print("\n[FINISHED] Analysis complete.")
print("Generated in:", OUT_DIR)
