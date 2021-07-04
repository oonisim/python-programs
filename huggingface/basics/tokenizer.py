#!/usr/bin/env python
# coding: utf-8

# # Tokenizer
# 
# * [Tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)
# 
# > A tokenizer is in charge of preparing the inputs for a model. 
# 
# ## Fast Tokenizer
# 
# Use **Fast** tokenizer., not the Python tokenizers.
# 
# > Most of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the Rust library tokenizers. 
# 
# ## Base Classes
# [PreTrainedTokenizerFast](https://huggingface.co/transformers/main_classes/tokenizer.html#pretrainedtokenizerfast) implements the common methods for encoding string inputs in inputs. Relies on PreTrainedTokenizerBase.
# 
# * **Tokenizing** <br>split strings in sub-word token strings, encode tokens into integer ids, decode ids back to tokens.
# * **Managing new tokens** <br>adding new tokens the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece…).
# 
# * **Managing special tokens** (mask, CLS/beginning-of-sentence, etc.)<br> adding and assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization.

# In[1]:


import json
import numpy as np
import tensorflow as tf
import transformers

from transformers import (
    TFDistilBertForSequenceClassification,
    TFAutoModelForSequenceClassification,
    DistilBertTokenizerFast
)


# # Configurations

# In[2]:


model_name = 'distilbert-base-uncased'
max_sequence_length = 256


# In[3]:


# model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    model_name, 
    truncation=True,
    padding=True,
    max_length=max_sequence_length,
    return_tensors="tf"
)


# # Tokenize
# 
# Note that you need to convert the result of the ```tokenizer``` which is ```transformers.tokenization_utils_base.BatchEncoding``` instance into dictionary to feed into the model.

# In[4]:


tokenized = tokenizer("A tokenizer is in charge of preparing the inputs for a model.")
print(type(tokenized))
print(tokenized)


# In[5]:


input = dict(tokenized)


# In[7]:


model(input)


# In[ ]:




