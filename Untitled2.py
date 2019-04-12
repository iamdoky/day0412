#!/usr/bin/env python
# coding: utf-8

# # 이거 뭐하는 거지...?
# 
# - 주피터 노트북에서 텐서플로우를 사용해본데...

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

a = tf.constant(100)
b = tf.constant(50)
add_op = a + b

v = tf.Variable(0)
let_op = tf.assign(v, add_op)


# In[2]:


sess = tf.Session()


# In[3]:


sess.run(tf.global_variables_initializer())


# In[4]:


sess.run(let_op)


# In[5]:


print(sess.run(v))


# In[ ]:




