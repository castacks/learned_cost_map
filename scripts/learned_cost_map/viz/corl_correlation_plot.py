import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style='white', font_scale=1.2)

df = pd.read_csv("/home/mateo/Data/SARA/TartanCost/freq_cost_functions.csv")
# df = df.rename(columns={'cost_1':'1-3 Hz',
#                    'cost_2':'1-5 Hz',
#                    'cost_3':'1-8 Hz',
#                    'cost_4':'1-12 Hz',
#                    'cost_5':'1-15 Hz',
#                    'cost_6':'1-20 Hz',
#                    'cost_7':'1-30 Hz',
#                    'cost_8':'3-5 Hz',
#                    'cost_9':'3-8 Hz',
#                    'cost_10':'3-12 Hz',
#                    'cost_11':'3-15 Hz',
#                    'cost_12':'3-20 Hz',
#                    'cost_13':'3-30 Hz',
#                    'cost_14':'5-8 Hz',
#                    'cost_15':'5-12 Hz',
#                    'cost_16':'5-15 Hz',
#                    'cost_17':'5-20 Hz',
#                    'cost_18':'5-30 Hz',
#                    'cost_19':'8-12 Hz',
#                    'cost_20':'8-15 Hz',
#                    'cost_21':'8-20 Hz',
#                    'cost_22':'8-30 Hz',
#                    'cost_23':'12-15 Hz',
#                    'cost_24':'12-20 Hz',
#                    'cost_25':'12-30 Hz',
#                    'cost_26':'15-20 Hz',
#                    'cost_27':'15-30 Hz',
#                    'cost_28':'20-30 Hz',
#                    })

sns.regplot(x=df["avg_score"], y=df["cost_7"], x_ci="sd").set(title="Correlation, Human vs Traversability Scores")  
# corrs = df.corr()

# plt.title("")
plt.xlabel("Average human traversability score")
plt.ylabel("Unnormalized traversability score")
plt.savefig('/home/mateo/corl_scores_correlation.png', dpi=300, bbox_inches="tight")
# plt.show()