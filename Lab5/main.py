#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import json
import operator
import numpy as np

from sklearn.decomposition import PCA

from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.models import Model
from keras.backend.tensorflow_backend import _to_tensor
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, TimeDistributed, Dense

from AttentionWithContext import AttentionWithContext


# In[2]:


path_root = ''
path_to_data = path_root + 'data/'

sys.path.insert(0, path_root)


# In[3]:


def bidir_gru(my_seq,n_units):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    '''
    
    # add a default GRU layer (https://keras.io/layers/recurrent/). You need to specify only the 'units' and 'return_sequences' arguments
    return Bidirectional(GRU(n_units, return_sequences=True),
                         merge_mode='concat', weights=None)(my_seq)


# In[4]:


# = = = = = parameters = = = = =

n_units   = 50
drop_rate = 0.5 
mfw_idx   = 2 # index of the most frequent words in the dictionary. 
              # 0 is for the special padding token
              # 1 is for the special out-of-vocabulary token

padding_idx  = 0
oov_idx      = 1
batch_size   = 32
nb_epochs    = 6
my_optimizer = 'adam'
my_patience  = 2 # for early stopping strategy


# In[5]:


# = = = = = data loading = = = = =

my_docs_array_train = np.load(path_to_data + 'docs_train.npy')
my_docs_array_test  = np.load(path_to_data + 'docs_test.npy')

my_labels_array_train = np.load(path_to_data + 'labels_train.npy')
my_labels_array_test  = np.load(path_to_data + 'labels_test.npy')

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open(path_to_data + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping
index_to_word = dict((v,k) for k,v in word_to_index.items())


# In[6]:


# = = = = = loading pretrained word vectors = = = = =

wvs = KeyedVectors.load(path_to_data + 'word_vectors.kv', mmap='r')

assert len(wvs.wv.vocab) == len(word_to_index) + 1 # vocab does not contain the OOV token

word_vecs = wvs.wv.syn0

pad_vec = np.random.normal(size=word_vecs.shape[1])

# add Gaussian vector on top of embedding matrix (padding vector)
word_vecs = np.insert(word_vecs,0,pad_vec,0)

print('embeddings created')

# reduce dimension with PCA (to reduce the number of parameters of the model)
my_pca = PCA(n_components=64)
embeddings_pca = my_pca.fit_transform(word_vecs)

print('embeddings compressed')


# In[7]:


# = = = = = defining architecture = = = = =

# = = = sentence encoder

sent_ints = Input(shape=(my_docs_array_train.shape[2],)) # vec of ints of variable size

sent_wv = Embedding(input_dim=embeddings_pca.shape[0], # vocab size
                    output_dim=embeddings_pca.shape[1], # dimensionality of embedding space
                    weights=[embeddings_pca],
                    input_length=my_docs_array_train.shape[2],
                    trainable=True
                    )(sent_ints)

sent_wv_dr = Dropout(drop_rate)(sent_wv)

# use bidir_gru, AttentionWithContext with return_coefficients=True, and Dropout
# warning: AttentionWithContext will return a list of two objects!
sent_gru = bidir_gru(sent_wv_dr, n_units)
sent_att_w_ctx, word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_gru)
sent_att_vec_dr = Dropout(drop_rate)(sent_att_w_ctx)

sent_encoder = Model(sent_ints,sent_att_vec_dr)


# In[8]:


# = = = document encoder

doc_ints = Input(shape=(my_docs_array_train.shape[1],my_docs_array_train.shape[2],))

### fill the gap (4 gaps) ###
# use TimeDistributed (https://keras.io/layers/wrappers/), bidir_gru, AttentionWithContext with return_coefficients=True, and Dropout
# warning: AttentionWithContext will return a list of two objects!

doc_td = TimeDistributed(sent_encoder)(doc_ints)
doc_gru = bidir_gru(doc_td, n_units)
doc_att_w_ctx, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_gru)
doc_att_vec_dr = Dropout(drop_rate)(doc_att_w_ctx)

preds = Dense(units=1, activation='sigmoid')(doc_att_vec_dr)

model = Model(doc_ints,preds)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

print('model compiled')


# In[ ]:


# = = = = = training = = = = =

loading_pretrained = False

if not loading_pretrained:
    
    # go through epochs as long as accuracy on validation set increases
    early_stopping = EarlyStopping(monitor='val_acc', patience=my_patience, mode='max')
    
    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'model', verbose=1, save_best_only=True, save_weights_only=True)
    
    # 200s/epoch on CPU - reaches 84.38% accuracy in 2 epochs
    model.fit(my_docs_array_train, 
              my_labels_array_train,
              batch_size = batch_size,
              epochs = nb_epochs,
              validation_data = (my_docs_array_test,my_labels_array_test),
              callbacks = [early_stopping,checkpointer])

else:
    model.load_weights(path_to_data + 'model')


# In[9]:


# = = = = = extraction of attention coefficients = = = = =

# define intermediate models: in each case, use the right inputs, and as outputs, the coefficients returned by the corresponding AttentionWithContext layer
get_word_att_coeffs = Model(sent_ints, word_att_coeffs) # extracts the attention coefficients of the words in a sentence
get_sent_att_coeffs = Model(doc_ints, sent_att_coeffs) # extracts the attention coefficients over the sentences in a document

my_review = my_docs_array_test[-1:,:,:] # select last review
# convert integer review to text
index_to_word[1] = 'OOV'
my_review_text = [[index_to_word[idx] for idx in sent if idx in index_to_word] for sent in my_review.tolist()[0]]


# In[10]:


# = = = attention over sentences in the document

sent_coeffs = get_sent_att_coeffs.predict(my_review)
sent_coeffs = sent_coeffs[0,:,:]

for elt in zip(sent_coeffs[:,0].tolist(),[' '.join(elt) for elt in my_review_text]):
    print(round(elt[0]*100,2),elt[1])


# In[11]:


# = = = attention over words in each sentence

my_review_tensor = _to_tensor(my_review,dtype='float32') # a layer, unlike a model, requires a TensorFlow tensor as input

# apply the 'get_word_att_coeffs' model over all the sentences in 'my_review_tensor'
word_coeffs = TimeDistributed(get_word_att_coeffs)(my_review_tensor)

word_coeffs = K.eval(word_coeffs) # shape = (7, 30, 1): (batch size, nb of sents in doc, nb of words per sent, coeff)

word_coeffs = word_coeffs[0,:,:,0] # shape = (7, 30) (coeff for each word in each sentence)

word_coeffs = sent_coeffs * word_coeffs # re-weigh according to sentence importance

word_coeffs = np.round((word_coeffs*100).astype(np.float64),2)

word_coeffs_list = word_coeffs.tolist()

# match text and coefficients
text_word_coeffs = [list(zip(words,word_coeffs_list[idx][:len(words)])) for idx,words in enumerate(my_review_text)]

for sent in text_word_coeffs:
    [print(elt) for elt in sent]
    print('= = = =')

# sort words by importance within each sentence
text_word_coeffs_sorted = [sorted(elt,key=operator.itemgetter(1),reverse=True) for elt in text_word_coeffs]

for sent in text_word_coeffs_sorted:
    [print(elt) for elt in sent]
    print('= = = =')

