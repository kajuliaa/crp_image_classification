import json
from scipy.stats import f_oneway
import matplotlib.pyplot as plt


with open("verystrong_relevance.json", "r") as f:
    verystrong_relevance = json.load(f)


with open("strong_relevance.json", "r") as f:
    strong_relevance = json.load(f)


with open("low_relevance.json", "r") as f:
    low_relevance = json.load(f)


with open("verylow_relevance.json", "r") as f:
    verylow_relevance = json.load(f)




# ANOVA-Test 
f_statistic, p_value = f_oneway(verystrong_relevance, strong_relevance, low_relevance, verylow_relevance)

print(f"F-Wert: {f_statistic:.4f}")
print(f"p-Wert: {p_value:.4f}")




data = [verystrong_relevance, strong_relevance, low_relevance, verylow_relevance]
labels = ['very strong', 'strong', 'low', 'very low']

plt.boxplot(data, labels=labels)
plt.ylabel('Explanation length')
plt.savefig("expl-length.png")
