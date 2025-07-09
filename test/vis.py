import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
data_dir = 'data'

train = os.path.join(data_dir, 'train_features.csv')
data = pd.read_csv(train)

print(data.head())
sns.countplot(data=data, x='label', order=data['label'].value_counts().index)
plt.show()