#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# here we define our document collection which contains 5 documents
# this is an array of strings
documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Bioinformatics studies the application of efficient algorithms in biological problems",
             "DNA sequences are very complex biological structures",
             "Genes are parts of a DNA sequence"]

tfidf_vectorizer = TfidfVectorizer()


# In[3]:


## TODO: get the matrix
tfidf_matrix_sparse = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix_sparse)


# In[4]:


# get the unique terms of the collection and display them
terms = tfidf_vectorizer.get_feature_names()

print("The unique terms of the collection are: ")
print(terms)

# print matrix dimensionality
print("The dimensionality of the tfidf matrix is: ")
print(tfidf_matrix_sparse.shape)

# print matrix contents


# In[5]:


## TODO: get the tabular form of the tf-idf matrix.
tfidf_matrix_dense = tfidf_matrix_sparse.todense()
print(tfidf_matrix_dense)


# In[6]:


## TODO: compute the doc-doc similarity matrix.
ddsim_matrix = cosine_similarity(tfidf_matrix_dense)

# define the doc-doc similarity matrix based on the cosine distance
print("This is the doc-doc similarity matrix :")
print(ddsim_matrix)
print("Diagonals of ones (since cosine between a document and itself is <X,X>/||X||^2 = 1)")
print("The matrix si simetric since cosine(X,Y) = cosine(Y,X)")


# In[7]:


# display the first line of the similarity matrix
# these are the similarity values between the first document with the rest of the documents
print("The first row of the doc-doc similarity matrix: ")
print(ddsim_matrix[:1])

cosine_1_2 = 0.42284413
angle_in_radians = math.acos(cosine_1_2)
angle_in_degrees = math.degrees(angle_in_radians)
print("The cosine of the angle between doc1 and doc2 is : \t" + str(cosine_1_2))
print("The angle (in radians) between doc1 and doc2 is  : \t"  + str(angle_in_radians))
print("The angle (in degrees) between doc1 and doc2 is  : \t"  + str(angle_in_degrees))

