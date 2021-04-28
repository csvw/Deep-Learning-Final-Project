import pandas as pd
import os
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join("/home/jupyter/data/", 'fixed_train.csv'))

plt.figure()
data.pivot_table(index=['class_name', "class_id"], aggfunc='size').plot(kind='bar')
plt.title("Class Imbalance")
plt.ylabel("Number of Instances")
plt.xlabel("Class")
plt.gcf().subplots_adjust(bottom=0.5)
plt.savefig("Classes.png")

print(data['class_name'].unique())

