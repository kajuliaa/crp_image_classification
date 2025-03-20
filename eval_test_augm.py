from scipy.stats import ttest_rel
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


with open("verystrong_relevance.json", "r") as f:
    verystrong_relevance = json.load(f)


with open("strong_relevance.json", "r") as f:
    strong_relevance = json.load(f)


with open("low_relevance.json", "r") as f:
    low_relevance = json.load(f)


with open("verylow_relevance.json", "r") as f:
    verylow_relevance = json.load(f)

#CHANGE NAME OF THE FILE
with open("verystrong_relevance_rot10.json", "r") as f:
    verystrong_relevance_augm = json.load(f)


with open("strong_relevance_rot10.json", "r") as f:
    strong_relevance_augm = json.load(f)


with open("low_relevance_rot10.json", "r") as f:
    low_relevance_augm = json.load(f)


with open("verylow_relevance_rot10.json", "r") as f:
    verylow_relevance_augm = json.load(f)



#paired T-test

t_stat, p_value = ttest_rel(verystrong_relevance, verystrong_relevance_augm)

print(f"t-Wert very strong: {t_stat:.4f}")
print(f"p-Wert very strong: {p_value:.4f}")


t_stat, p_value = ttest_rel(strong_relevance, strong_relevance_augm)

print(f"t-Wert strong: {t_stat:.4f}")
print(f"p-Wert strong: {p_value:.4f}")


t_stat, p_value = ttest_rel(low_relevance, low_relevance_augm)

print(f"t-Wert low: {t_stat:.4f}")
print(f"p-Wert low: {p_value:.4f}")

t_stat, p_value = ttest_rel(verylow_relevance, verylow_relevance_augm)

print(f"t-Wert very low: {t_stat:.4f}")
print(f"p-Wert very low: {p_value:.4f}")






#PLOTTING THE RESULTS

def create_df(original, rotated, label):
    return pd.DataFrame({
        'Image': ['Original']*len(original) + ['Rotation']*len(rotated),
        'Explanation Length': original + rotated
    })


# Plot 1: Very Strong
df_vs = create_df(verystrong_relevance, verystrong_relevance_augm, 'Very Strong')
plt.figure(figsize=(5, 4))
sns.boxplot(x='Image', y='Explanation Length', data=df_vs, palette='Set2')
plt.title('Very Strong Relevance')
plt.tight_layout()
plt.savefig("verystrong-relevance-rot10.png")

# Plot 2: Strong
df_s = create_df(strong_relevance, strong_relevance_augm, 'Strong')
plt.figure(figsize=(5, 4))
sns.boxplot(x='Image', y='Explanation Length', data=df_s, palette='Set2')
plt.title('Strong Relevance')
plt.tight_layout()
plt.savefig("strong-relevance-rot10.png")

# Plot 3: Low
df_l = create_df(low_relevance, low_relevance_augm, 'Low')
plt.figure(figsize=(5, 4))
sns.boxplot(x='Image', y='Explanation Length', data=df_l, palette='Set2')
plt.title('Low Relevance')
plt.tight_layout()
plt.savefig("low-relevance-rot10.png")


# Plot 4: Very Low
df_vl = create_df(verylow_relevance, verylow_relevance_augm, 'Very Low')
plt.figure(figsize=(5, 4))
sns.boxplot(x='Image', y='Explanation Length', data=df_vl, palette='Set2')
plt.title('Very Low Relevance')
plt.tight_layout()
plt.savefig("verylow-relevance-rot10.png")





