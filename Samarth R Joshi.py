#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


mcdonalds = pd.read_csv("https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv")

# Display column names
print(mcdonalds.columns)

# Display dimensions
print(mcdonalds.shape)

# Display the first 3 rows
print(mcdonalds.head(3))


# In[ ]:





# In[3]:


# Select columns 1 to 11 and convert "Yes" to 1 and "No" to 0
MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == "Yes").astype(int))

# Calculate and round column means
means = MD_x.mean().round(2)

# Display the result
print(means)


# In[5]:


from sklearn.decomposition import PCA


# Perform PCA
MD_pca = PCA()
MD_pca.fit(MD_x)

# Display summary
print("Importance of components:")
print("\n".join([f"PC{i} {s:.4f}" for i, s in enumerate(MD_pca.explained_variance_ratio_, 1)]))

print("\nCumulative Proportion:")
print("\n".join([f"PC{i} {sum(MD_pca.explained_variance_ratio_[:i]):.4f}" for i in range(1, len(MD_pca.explained_variance_ratio_)+1)]))


# In[6]:


print("Standard deviations (1, .., p=11):")
print(MD_pca.explained_variance_**0.5)

# Display rotation matrix
print("Rotation (n x k) = (11 x 11):")
print(pd.DataFrame(MD_pca.components_, columns=MD_x.columns).round(2))


# In[8]:


plt.scatter(MD_pca.transform(MD_x)[:, 0], MD_pca.transform(MD_x)[:, 1], color='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')
plt.show()

# Display projected axes
print("Projected Axes:")
print(pd.DataFrame(MD_pca.components_[:2, :].T, index=MD_x.columns, columns=['PC1', 'PC2']).round(2))


# In[ ]:





# In[ ]:





# In[ ]:





# # Using Mixtures of Regression Models

# In[26]:


import pandas as pd

# Assuming you have a DataFrame named "mcdonalds" with the column "Like"
# Replace this with your actual data
mcdonalds = pd.DataFrame({"Like": ["I hate it!"] * 152 + ["-4"] * 71 + ["-3"] * 73 + ["-2"] * 59 +
                                  ["-1"] * 58 + ["0"] * 169 + ["+1"] * 152 + ["+2"] * 187 +
                                  ["+3"] * 229 + ["+4"] * 160 + ["I love it!"] * 143})

# Reverse the order of "Like" and create a new variable "Like.n"
mcdonalds['Like.n'] = 6 - mcdonalds['Like'].apply(lambda x: int(x) if x.lstrip('-').isdigit() else 0)

# Display the table for the new variable "Like.n"
print(mcdonalds['Like.n'].value_counts().sort_index())


# # Profiling Segments

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Assuming MD.x is a numpy array or a pandas DataFrame
MD_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Perform hierarchical clustering
MD_vclust = linkage(pdist(MD_x.T), method='average')

# Create the segment profile plot
plt.figure(figsize=(10, 6))

# Plot the dendrogram
dendrogram(MD_vclust, labels=range(1, MD_x.shape[1] + 1))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Segment Variables')
plt.ylabel('Cluster Distance')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:




