import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df=pd.read_csv("ki.csv")
df_encoded=pd.get_dummies(df)
frequent_itemsets=apriori(df_encoded,min_support=0.05,use_colnames=True)
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1.0)
print("association rules")
print(rules)
