#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
from nltk.corpus import stopwords
from library import clean_text_simple, terms_to_graph, unweighted_k_core

# execute the following if you haven't already (nltk > 3.2.1 is required)
#import nltk 
#nltk.download('stopwords')
#nltk.download('maxent_treebank_pos_tagger')
#nltk.download('averaged_perceptron_tagger')

#import os
#os.chdir() # change working directory to where functions are


# In[2]:


my_doc = '''A method for solution of systems of linear algebraic equations with m-dimensional lambda matrices. A system of linear algebraic equations with m-dimensional lambda matrices is considered. The proposed method of searching for the solution of this system lies in reducing it to a numerical system of a special kind.'''
#my_doc = my_doc.replace('\n', '')
print(my_doc)


# In[3]:


# pre-process document
stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)
print("Number of different tokens =", len(set(my_tokens)))


# In[4]:


# build the graph
g = terms_to_graph(my_tokens, w=4)
print("Number of vertices = ", len(g.vs))
print("Number of edges    = ", len(g.es))
assert len(g.vs) == len(set(my_tokens)) # the number of nodes should be equal to the number of unique terms

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)


# In[5]:


# build a graph-of-words g
for w in range(2,min(len(my_tokens)+1,30)):
    g = terms_to_graph(my_tokens, w=w)
    print(g.density())

print("""\nSliding window size increases => Density increases. 
It is never reaching 1 simply because edges are weighted
(some pairs of unique words can appear together in multiple windows)""")


# In[6]:


# decompose g
core_numbers = unweighted_k_core(g)
print(core_numbers)

# compare with igraph method
print(dict(zip(g.vs['name'],g.coreness())))


# In[7]:


# retain main core as keywords
max_c_n = max(list(core_numbers.values()))
keywords = [key for key, core in core_numbers.items() if core == max_c_n]
print(keywords)

