import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_excel(r"C:\Users\Andre.Vieira\Downloads\Teste\registros-prod.xlsx")

#print(df.head())
#print('\n')
#print(df.info())
#print('\n')
#print(df.describe())
#print('\n')
#print(df.VAR_2.unique())
#print(df.VAR_2.value_counts())

# Compute the correlation matrix
corr = df.corr()

print(corr)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#cm = confusion_matrix(corr, predictions, labels=clf.classes_)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                              display_labels=clf.classes_)
#disp.plot()

#plt.show()