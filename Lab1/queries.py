#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


documents = ["Euler is the father of graph theory",
             "Graph theory studies the properties of graphs",
             "Bioinformatics studies the application of efficient algorithms in biological problems",
             "DNA sequences are very complex structures",
             "Genes are parts of a DNA sequence",
             "Run to the hills, run for your lives",
             "The lonenliness of the long distance runner",
             "Heaven can wait til another day",
             "Road runner and coyote is my favorite cartoon",
             "Heaven can wait"] # the last document is our query (orignially "Heaven can can Heaven can graph" but changed to "" to match the lab sheet)


# In[3]:


def my_cosine_similarity(vector1, vector2):
    ## TODO: complete the lines as given in the description
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if magnitude:
        return dot_product/magnitude
    return 0


# In[4]:


tfidf_vectorizer = TfidfVectorizer()
## TODO: get the tabular form of the tf-idf matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(documents).todense().A # N


# In[5]:


print("Similarity among the query and the documents: ")
for x in range(9):
    ## TODO: use the my_cosine_similarity
    print("     cosine({},query) = {}".format(x,my_cosine_similarity(tfidf_matrix[x,:],tfidf_matrix[9,:])))


# In[6]:


print("The query vector is of len {}, which is the total number of different words in the documents".format(len(tfidf_matrix[9,:])))
print("""The only non-zero similarity (with vector 7) comes from the fact that document 7 is the only one
 to have words in common with query. Actually it has 3 of them (Heaven, can, wait), which btw are
 all the words of the query.""")
      

