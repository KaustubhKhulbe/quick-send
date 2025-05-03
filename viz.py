import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean
df = pd.read_csv("consolidated_output.csv")
df.dropna(inplace=True)

numeric_cols = ["Cache Size", "Bytes Sent", "Total Bytes", 
                "Compression Ratio", "Hits", "Misses"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# --- Compression Ratio vs Cache Size (by Algorithm only) ---
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="Cache Size", y="Compression Ratio", hue="Algorithm", markers=True, dashes=False)
plt.title("Compression Ratio vs Cache Size (Grouped by Algorithm)")
plt.tight_layout()
plt.savefig("compression_ratio_vs_cache_size.png")
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# --- Individual Compression Ratio Distribution Plots ---
unique_algorithms = df["Algorithm"].unique()
for alg in unique_algorithms:
    subset = df[df["Algorithm"] == alg]
    plt.figure(figsize=(7, 5))
    sns.histplot(data=subset, x="Compression Ratio", kde=True, bins=20)
    plt.title(f"Compression Ratio Distribution - {alg}")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"compression_ratio_distribution_{alg}.png")
    plt.close()

# --- Bar Plots: Avg Compression Ratio per Pattern for Each Algorithm ---
algorithms = df["Algorithm"].unique()

for alg in algorithms:
    subset = df[df["Algorithm"] == alg]
    avg_by_pattern = subset.groupby("Pattern")["Compression Ratio"].mean().reset_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=avg_by_pattern, x="Pattern", y="Compression Ratio", palette="muted")

    # Annotate bars with values
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)

    plt.title(f"Average Compression Ratio by Pattern - {alg}")
    plt.ylabel("Avg Compression Ratio")
    
    # Set y-axis range close to the data (with small buffer)
    min_y = avg_by_pattern["Compression Ratio"].min()
    max_y = avg_by_pattern["Compression Ratio"].max()
    plt.ylim(min_y - 0.05, max_y + 0.1)

    plt.tight_layout()
    plt.savefig(f"avg_compression_ratio_by_pattern_{alg}.png")
    plt.close()

# --- Heatmaps: Compression Ratio by Pattern and Cache Size for Each Algorithm ---
algorithms = df["Algorithm"].unique()

for alg in algorithms:
    subset = df[df["Algorithm"] == alg]

    # Pivot for heatmap: rows = Pattern, cols = Cache Size, values = Compression Ratio
    heatmap_data = subset.pivot_table(index="Pattern", columns="Cache Size", values="Compression Ratio", aggfunc="mean")

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, linecolor='white')

    plt.title(f"Compression Ratio Heatmap\nAlgorithm: {alg}")
    plt.xlabel("Cache Size")
    plt.ylabel("Pattern")
    plt.tight_layout()
    plt.savefig(f"compression_ratio_heatmap_{alg}.png")
    plt.close()

# --- Standard Deviation of Compression Ratio by Pattern ---
pattern_var = df.groupby(["Algorithm", "Pattern"])["Compression Ratio"].std().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=pattern_var, x="Pattern", y="Compression Ratio", hue="Algorithm", palette="Set2")
plt.title("Compression Ratio Variability by Pattern")
plt.ylabel("Standard Deviation")
plt.xlabel("Pattern")
plt.tight_layout()
plt.savefig("compression_ratio_variability_by_pattern.png")
plt.close()


# --- Standard Deviation of Compression Ratio by Cache Size ---
cache_var = df.groupby(["Algorithm", "Cache Size"])["Compression Ratio"].std().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=cache_var, x="Cache Size", y="Compression Ratio", hue="Algorithm", marker="o", palette="Set1")
plt.title("Compression Ratio Variability by Cache Size")
plt.ylabel("Standard Deviation")
plt.xlabel("Cache Size")
plt.tight_layout()
plt.savefig("compression_ratio_variability_by_cache_size.png")
plt.close()

# --- Box Plot: Compression Ratio Distribution by Pattern ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Pattern", y="Compression Ratio", hue="Algorithm", palette="pastel")
plt.title("Compression Ratio Distribution by Pattern")
plt.ylabel("Compression Ratio")
plt.xlabel("Pattern")
plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("compression_ratio_distribution_by_pattern.png")
plt.close()

# --- Box Plot: Compression Ratio Distribution by Cache Size ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Cache Size", y="Compression Ratio", hue="Algorithm", palette="colorblind")
plt.title("Compression Ratio Distribution by Cache Size")
plt.ylabel("Compression Ratio")
plt.xlabel("Cache Size")
plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("compression_ratio_distribution_by_cache_size.png")
plt.close()

# --- Strip Plot: Compression Ratio Comparison by Pattern ---
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x="Pattern", y="Compression Ratio", hue="Algorithm", jitter=True, palette="Set1")
plt.title("Compression Ratio Comparison by Pattern")
plt.tight_layout()
plt.savefig("compression_ratio_comparison_by_pattern.png")
plt.close()

# --- Boxplot for Cache Size vs Compression Ratio ---
df["Efficiency"] = df["Compression Ratio"]

# Remove outliers (IQR method) within each group
df_filtered = pd.DataFrame()

for (alg, cache), group in df.groupby(["Algorithm", "Cache Size"]):
    Q1 = group["Efficiency"].quantile(0.25)
    Q3 = group["Efficiency"].quantile(0.75)
    IQR = Q3 - Q1
    filtered_group = group[(group["Efficiency"] >= Q1 - 1.5 * IQR) & (group["Efficiency"] <= Q3 + 1.5 * IQR)]
    df_filtered = pd.concat([df_filtered, filtered_group], ignore_index=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Cache Size", y="Efficiency", hue="Algorithm", palette="dark", showfliers=False)
plt.title("Compression Efficiency by Cache Size")
plt.ylabel("Compression Efficiency")
plt.xlabel("Cache Size")
plt.tight_layout()
plt.savefig("compression_efficiency_by_cache_size.png")
plt.close()
