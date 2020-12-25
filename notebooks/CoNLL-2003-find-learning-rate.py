#!/usr/bin/env python
# coding: utf-8

# ## Learning rate search for Conll-2003

# In[ ]:


get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import sys; sys.path.append("..")
import matplotlib.pyplot as plt
import importlib

import pycrf.train
importlib.reload(pycrf.train)
from pycrf.train import Learner

# In[ ]:


learner = Learner.build(train="../data/conll2003/train.feats.txt",
                        validation="../data/conll2003/valid.feats.txt",
                        word_vectors="../data/glove.6B.50d.txt",
                        verbose=True,
                        cuda=True)


# In[ ]:


lrs, losses = learner.find_lr()


# In[ ]:


fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(lrs, losses)
ax.grid()

