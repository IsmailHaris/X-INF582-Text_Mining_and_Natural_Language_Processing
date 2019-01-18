#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# here we define our document collection which contains 5 documents
# this is an array of strings
documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Graph theory is cool!",
             "DNA sequences are very complex biological structures",
             "Genes are biological structures that are parts of a DNA sequence",
             "Genes are very important biological structures"]

# create the tf-idf vectors of the document collection
tfidf_vectorizer = TfidfVectorizer()


# In[3]:


## TODO: get the matrix
A = tfidf_vectorizer.fit_transform(documents).todense()


# In[4]:


# apply Singular Value Decomposition
U, S, V = np.linalg.svd(A)
print(U.shape)
print(S.shape)
print(V.shape)

print(S)

print(V.transpose())
# this is the original matrix
print(A)

# keep the first two rows of V
V2 = V[:2,:]
print(V2)


# In[5]:


# the matrix after dimensionality reduction
## TODO: get the M matrix
M = np.dot(A, V.transpose())
print(M)


# In[6]:


# plot the results
colors = ['blue','red','black','green','orange','brown']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(6):
    ax.scatter(M[i,0],M[i,1], color=colors[i])
ax.scatter(0,0,color='black')
plt.xlabel('SVD1')
plt.xlabel('Concept 1')
plt.ylabel('Concept 2')
plt.show()

print("""Comment: We can see that 2 clusters emerge, surely corresponding to the 2 topics of Graph Theroy and Bioinformatics""")


# In[ ]:




