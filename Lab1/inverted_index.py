#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Inverted Index demonstration:
- index construction from a text collection
- query the index

"""

import re
import glob


# In[2]:


'''
For each file create a list of terms, ignoring punctuation and converting upper case letters
to lower case.
'''
def processFiles(filenames):
    terms_of_file = {}
    pattern = re.compile('[\W_]+')
    for file in filenames:
        terms_of_file[file] = open(file,'r', encoding="utf-8").read().lower()
        terms_of_file[file] = pattern.sub(' ', terms_of_file[file])
        re.sub(r'[\W_]+', '', terms_of_file[file])
        terms_of_file[file] = terms_of_file[file].split()
    return terms_of_file


'''
For the input file create the positional list: a dictionary where the key is the word and
the value is a list containing the positions where the word appears in the corresponding
document.
'''
def indexOneFile(termlist):
    fileIndex = {}
    for word in termlist:
        fileIndex[word] = [i for i, e in enumerate(termlist) if e == word]
    return fileIndex

'''
Create a dictionary where the keys are the filenames and the values are the positional lists
of the files.
'''
def buildIndexPerFile(termlists):
    total = {}
    for filename, termlist in termlists.items():
        total[filename] = indexOneFile(termlist)
    return total

'''
Use the index_per_file and create the inverted index of the collection. The inverted index is
a dictionary, where the key is the word and the value is the positional list of the word in all
documents that it is contained in.
'''
def buildFullIndex(index_per_file):
    total_index = {}
    for filename in list(index_per_file.keys()):
        for word in list(index_per_file[filename].keys()):
            if word in list(total_index.keys()):
                if filename in list(total_index[word].keys()):
                    total_index[word][filename].extend(index_per_file[filename][word][:])
                else:
                    total_index[word][filename] = index_per_file[filename][word]
            else:
                total_index[word] = {filename: index_per_file[filename][word]}
    return total_index

'''
Use the inverted index to process a query composed of a single term only. This will be used
as a building block to answer multi-term queries.
'''
def singleTermQuery(word, inverted_index):
    pattern = re.compile('[\W_]+')
    word = pattern.sub(' ',word)
    if word in list(inverted_index.keys()):
        return [filename for filename in list(inverted_index[word].keys())]
    else:
        return []

'''
Use the inverted index to find the union of the inverted lists for documents that contain at
least one term of the query.
'''
def multiTermQuery(string, inverted_index):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    result = []
    for word in string.split():
        result += singleTermQuery(word, inverted_index)
    return list(set(result))

'''
Execute a phrase-based query, i.e., all words must appear in the document in the corresponding
order.
'''
def phraseQuery(string, inverted_index):
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    listOfLists, result = [],[]
    for word in string.split():
        listOfLists.append(singleTermQuery(word, inverted_index))
    setted = set(listOfLists[0]).intersection(*listOfLists)
    for filename in setted:
        temp = []
        for word in string.split():
            temp.append(inverted_index[word][filename][:])
        for i in range(len(temp)):
            for ind in range(len(temp[i])):
                temp[i][ind] -= i
        if set(temp[0]).intersection(*temp):
            result.append(filename)
    return result


# In[3]:


'''
Populate list with filenames of the document collection.
'''
docnames = [file for file in glob.glob("f*.txt")]
print('Filenames of the document collection: ')
print(docnames)
print()

tof = processFiles(docnames)
for item in list(tof.keys()):
    print(str(item) + '  ' + str(tof[item]))
print()


# In[4]:


'''
Preprocess document terms, e.g., stopword removal, stemming etc
'''

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


for item in list(tof.keys()):
    tof[item] = [w for w in tof[item] if w not in stopwords]


# In[5]:


'''
Build the inverted index
'''

total = buildIndexPerFile(tof)
print(total)
print()

inverted_index = buildFullIndex(total)


# In[6]:


'''
Sort terms in lexicographic order (for display purposes only)
'''
sorted_terms = sorted(inverted_index.keys())


# In[7]:


'''
Display the inverted index
'''
print('****************************************************************************')
print('************* PART OF THE INVERTED INDEX FOR THE COLLECTION ****************')
print('****************************************************************************')
print()
for term in sorted_terms:
    if term >= "w":
        print(str(term) + ' ----> ' + str(inverted_index[term]))
print()


# In[8]:


print('************************************************************')

print("{:12s} appear in {}".format("'would'", multiTermQuery("would", inverted_index)))
print("{:12s} appear in {}".format("'years'", multiTermQuery("years", inverted_index)))
print("{:12s} appear in {}".format("'many' or 'years'", multiTermQuery("many years", inverted_index)))
print()
print("{:12s} appear in {}".format("'monetary crisis'", phraseQuery("monetary crisis", inverted_index)))


# In[ ]:




