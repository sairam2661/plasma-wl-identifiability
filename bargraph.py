import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data = pd.read_csv("data/log_data/ModifiedFiles/12.csv")

rank = data["WilCoxnRank-w"].to_list()
Peptide = data["Peptidoform"].to_list()
data = {
    "WilCoxnRank-w": rank,
    "Peptidoform": Peptide
}

# Create DataFrame and sort by Peptidoform (optional)
df = pd.DataFrame(data)
df_sorted = df.sort_values(by="WilCoxnRank-w", ascending=False)

# Generate colorful bars using a colormap
colors = plt.cm.rainbow(np.linspace(0, 1, len(df_sorted)))

# Plot
plt.figure(figsize=(20, 10))
bars = plt.bar(df_sorted["Peptidoform"], df_sorted["WilCoxnRank-w"], color=colors, edgecolor='black')

# Add labels
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("WilCoxnRank-w", fontsize=14)
plt.xlabel("Peptidoform", fontsize=14)
plt.title("Bar Chart: WilCoxnRank-w vs. Peptidoform", fontsize=18, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 40:  # Only label taller bars for clarity
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{height:.1f}', 
                 ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
# plt.show()

plt.savefig("data-6.png")
