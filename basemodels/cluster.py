# %%
import pandas as pd 
import random 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
import pickle

# %%
df =  pd.read_csv("../dataset/processed_data/shu_old_fade_rate.csv",index_col=0)
#df = pd.read_csv("../dataset/processed_data/nature_processed_dup.csv",index_col=0)
cols = list(df.columns)

# %%
labels  = []

for idx,col in enumerate(df.columns): 



    if idx < 9:
        plt.plot(df[col])
        labels.append(col)
    else: 
        plt.plot(df[col])

labels.append("……")
plt.legend(labels,ncol=2) 

plt.xlabel("Cycle Number")
plt.ylabel("SOC")


plt.show()

# %%
# 随机选择
while True: 
    test_cols = random.sample(cols,k=6)
    train_cols = [ x for x in cols if x not in test_cols ]

    flag = False

    for col in train_cols: 
        if len(list(df[col].dropna())) < 400:
            flag = True

    if not flag: 
        break            

# %%

first_list =  {}
second_list = {}
third_list = {}
third2_list = {}
third3_list = {}

for col in df.columns:

    length_of_col =  len(df[col].dropna())
    small_value = int(length_of_col * 0.2 )

    if small_value < 100:
        print("temp ")
        continue
    mid_value = int(length_of_col * 0.4 ) 
    
    mid2_value = int(length_of_col * 0.6 ) 
    
    mid3_value = int(length_of_col * 0.8 ) 
    
    print("lnn: " + str(small_value))
    
    first_list[col] = pd.Series(df[col].dropna().to_list()[:small_value])
    second_list[col] =   pd.Series(df[col].dropna().to_list()[small_value: mid_value])
    third_list[col] = pd.Series(df[col].dropna().to_list()[mid_value:mid2_value ])
    third2_list[col] = pd.Series(df[col].dropna().to_list()[mid2_value:mid3_value ])
    third3_list[col] = pd.Series(df[col].dropna().to_list()[mid3_value: ])
    
    
pd.DataFrame().from_dict(first_list).to_csv("../dataset/processed_data/shu_old_fade_rate_small.csv")
pd.DataFrame().from_dict(second_list).to_csv("../dataset/processed_data/shu_old_fade_rate_mid.csv")
pd.DataFrame().from_dict(third_list).to_csv("../dataset/processed_data/shu_old_fade_rate_mid2.csv")
pd.DataFrame().from_dict(third2_list).to_csv("../dataset/processed_data/shu_old_fade_rate_mid3.csv")
pd.DataFrame().from_dict(third3_list).to_csv("../dataset/processed_data/shu_old_fade_rate_large.csv")

# %%
df[col].dropna()[:small_value]

# %%
bucket_list = []

battery_list = []

for col in train_cols:

    battery_list.append(list(df[col].dropna())[:400])

x = np.array(battery_list)
cluster = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(x)

labels = cluster.labels_
colors = ['r','b','g','black',"y","brown"]

plt.title("Clusters")

my_labels=["Cluster_0","Cluster_1","Cluster_2"]

for label in range(3):
    cur_bucket = [ col for idx,col in  enumerate(train_cols)  if labels[idx] == label  ]

    print(len(cur_bucket))

    bucket_list.append(cur_bucket)

    
    plt.plot(df[cur_bucket[1:]],color=colors[label])
    plt.plot(df[cur_bucket[0]],color=colors[label],label=my_labels[label])

plt.xlabel("Cycle Number")
plt.ylabel("SOC")


plt.legend()


plt.show()


# %%
with open("./args/bucket_list.pickle","wb")  as handler:
    pickle.dump(bucket_list,handler)

# %%
with open("./args/bucket_list.pickle","rb")  as handler:
    bucket_list = pickle.load(handler)

# %%
with open("./args/train_args.pkl","wb") as handler: 
    pickle.dump((train_cols,test_cols),handler)

# %%
with open("./args/train_args.pkl","rb") as f:
    (train_all_cols,test_cols) = pickle.load(f)
    

# %%
with open("./args/train_args.pkl","rb") as f:
    (train_all_cols,test_cols) = pickle.load(f)


