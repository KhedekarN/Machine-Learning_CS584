#!/usr/bin/env python
# coding: utf-8

# In[25]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 1a) Create a dataset which contains the number of distinct items in each customer’s market basket. Draw a histogram of the number of unique items.  What are the median, the 25th percentile and the 75th percentile in this histogram?

# In[26]:


df1 = pd.read_csv(r'/kaggle/input/Groceries.csv',delimiter=',', usecols=['Customer', 'Item'])


# In[27]:


distinctItemSet= pd.DataFrame(df1["Item"].unique(),columns=["Itemsets"])
print("Number of distinct itemsets is",len(df1["Item"].unique()))
distinctItemSet.head()


# In[28]:


ListOfItems = df1.groupby(['Customer'])['Item'].apply(list).values.tolist()
NumberOfCustomers = len(ListOfItems)
print("Number of Customers = ", NumberOfCustomers)


# In[78]:


list1=[]
for x in ListOfItems:
    list1.append(len(x))
plt.hist(list1, bins=32)
plt.xlabel('Number of unique items per customer ')
plt.ylabel('Number of customers having respective number of items')


# In[30]:


df2=pd.DataFrame(list1)
print("median, the 25th percentile and the 75th percentile in above histogram is")
df2.describe()


# 1b)If you are interested in the k-itemsets which can be found in the market baskets of at least seventy five (75) customers.  How many itemsets can you find?  Also, what is the largest k value among your itemsets?

# In[31]:


# Finding the frequent itemsets
te = TransactionEncoder()
te_ary = te.fit(ListOfItems).transform(ListOfItems)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

# Calculating the frequency table
frequent_itemsets = apriori(ItemIndicator, min_support = 75/9835, use_colnames = True)
frequent_itemsets.head()


# In[83]:


# Find the frequent itemsets
te = TransactionEncoder()
te_ary = te.fit(ListOfItems).transform(ListOfItems)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

# Calculate the frequency table
frequent_itemsets = apriori(ItemIndicator, min_support = 75/9835, use_colnames = True)
frequent_itemsets.head()
print("Number of frequent Itemsets:",len(frequent_itemsets))


# In[33]:


list2 = []
for i in range(0,len(frequent_itemsets["itemsets"])):
    list2.append(len(frequent_itemsets["itemsets"][i]))
print("Largest k value among the itemset is",max(list2))


# 1c) Find out the association rules whose Confidence metrics are at least 1%.  How many association rules have you found?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.  Also, you do not need to show those rules.

# In[34]:


assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("The total number of association rules are",len(assoc_rules))


# 1d)Graph the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you found in (c).  Please use the Lift metrics to indicate the size of the marker. 

# In[35]:


plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()


# 1e) List the rules whose Confidence metrics are at least 60%.  Please include their Support and Lift metrics.

# In[36]:


assoc_rules_60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
assoc_rules_60


# 2a)Generate a scatterplot of y (vertical axis) versus x (horizontal axis).  How many clusters will you say by visual inspection?

# In[37]:


#question 2
df4 = pd.read_csv("/kaggle/input/Spiral.csv")
df4.head()


# In[85]:


plt.scatter(df4.x,df4.y)
plt.xlabel('X')
plt.ylabel('Y')


# 2b) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[39]:


import sklearn.cluster as cluster
Tdata=df4[["x","y"]]

kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(Tdata)

print("Cluster Assignment:", kmeans.labels_)
print("Cluster Centroid 0:", kmeans.cluster_centers_[0])
print("Cluster Centroid 1:", kmeans.cluster_centers_[1])


# In[40]:



df4["Cluster_Labels"]= kmeans.labels_
sns.scatterplot(df4.x, df4.y, hue=df4.Cluster_Labels)


# 2c)	(10 points) Apply the nearest neighbor algorithm using the Euclidean distance.  How many nearest neighbors will you use?  Remember that you may need to try a couple of values first and use the eigenvalue plot to validate your choice.

# In[109]:


import sklearn.neighbors

from sklearn.neighbors import NearestNeighbors as kNN
kNNSpec = kNN(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(Tdata)
distances, indices = nbrs.kneighbors(Tdata)

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(Tdata)


# In[110]:


import math
nObs = df4.shape[0]
# Create the Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in indices[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)


# In[111]:



plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()


# 2d)	(10 points) Retrieve the first two eigenvectors that correspond to the first two smallest eigenvalues.  Display up to ten decimal places the means and the standard deviation of these two eigenvectors.  Also, plot the first eigenvector on the horizontal axis and the second eigenvector on the vertical axis.

# In[118]:



# Inspect the values of the selected two eigenvectors 
Z = evecs[:,[0,1]]
print("1st eigenvectors:",Z[[0]])
print("2nd eigenvectors:",Z[[1]])
plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()


# In[120]:


meanz0= Z[[0]].mean()
print("mean of 1st eigenvector",round(meanz0,10))
meanz1= Z[[1]].mean()
print("mean of 2nd eigenvector",round(meanz1,10))
stdz0 = Z[[0]].std()
print("standard deviation of 1st eigenvector",round(stdz0,10))
stdz1= Z[[1]].std()
print("standard deviation of 2nd eigenvector",round(stdz1,10))


# 2e)	(10 points) Apply the K-mean algorithm on your first two eigenvectors that correspond to the first two smallest eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[64]:


kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)Z
df4['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(df4['x'], df4['y'], c = df4['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[74]:




