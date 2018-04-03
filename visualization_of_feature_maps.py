# coding: utf-8
import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
train = mnist.test.images
im = np.reshape(train[73],(28,28))  
fig = plt.figure(1)  
plt.imshow(im,cmap='gray')
plt.axis('off')
plt.savefig('/home/liuyang/1.eps',format='eps',dpi=500)

# In[36]:
#base1
get_ipython().magic('matplotlib inline')
import os
import numpy as np  
import pickle as pk
import matplotlib.pyplot as plt 
home='/home/liuyang'
def save_(n):
    fig = plt.figure(1)
    if os.path.isdir(home+'/save/base/%d'%(n)):
        f=open('/home/liuyang/save/base/%d/feature_map1.pk'%n,'rb')
        fm=np.reshape(pk.load(f),(28,28,32))
        f.close()
        for i in range(32):
            plt.subplot(4,8,i+1)
            plt.imshow(fm[:,:,i],cmap='gray')
            plt.axis('off')
        plt.savefig('/home/liuyang/save/base/%d/fm1.eps'%n,format='eps',dpi=1000)
def save_1(n):
    fig = plt.figure(2)
    if os.path.isdir(home+'/save/base/%d'%(n)):
        f=open('/home/liuyang/save/base/%d/feature_map2.pk'%n,'rb')
        fm=np.reshape(pk.load(f),(14,14,64))
        f.close()
        j=0
        for i in range(64):
            if (i<=7)|((i>=16)&(i<=23))|((i>=32)&(i<=39))|((i>=48)&(i<=55)):
                j+=1
                plt.subplot(4,8,j)
                plt.imshow(fm[:,:,i],cmap='gray')
                plt.axis('off')
        plt.savefig('/home/liuyang/save/base/%d/fm21.eps'%n,format='eps',dpi=1000)
def save_2(n):
    fig = plt.figure(2)
    if os.path.isdir(home+'/save/base/%d'%(n)):
        f=open('/home/liuyang/save/base/%d/feature_map2.pk'%n,'rb')
        fm=np.reshape(pk.load(f),(14,14,64))
        f.close()

        for i in range(8):
            x=(fm[:,:,i]+fm[:,:,i+16]+fm[:,:,i+32]+fm[:,:,i+48])/10.0
            plt.subplot(1,8,i+1)
            plt.imshow(x,cmap='gray')
            plt.axis('off')
        plt.savefig('/home/liuyang/save/base/%d/fm_superposition1.eps'%n,format='eps',dpi=1000)
def save_3(n):
    fig = plt.figure(2)
    if os.path.isdir(home+'/save/base/%d'%(n)):
        f=open('/home/liuyang/save/base/%d/feature_map2.pk'%n,'rb')
        fm=np.reshape(pk.load(f),(14,14,64))
        f.close()
        j=0
        for i in range(64):
            if ((i>=8)&(i<=15))|((i>=24)&(i<=31))|((i>=40)&(i<=47))|((i>=56)&(i<=63)):
                j+=1
                plt.subplot(4,8,j)
                plt.imshow(fm[:,:,i],cmap='gray')
                plt.axis('off')
        plt.savefig('/home/liuyang/save/base/%d/fm22.eps'%n,format='eps',dpi=1000)
def save_4(n):
    fig = plt.figure(2)
    if os.path.isdir(home+'/save/base/%d'%(n)):
        f=open('/home/liuyang/save/base/%d/feature_map2.pk'%n,'rb')
        fm=np.reshape(pk.load(f),(14,14,64))
        f.close()

        for i in range(8,16):
            x=(fm[:,:,i]+fm[:,:,i+16]+fm[:,:,i+32]+fm[:,:,i+48])/10.0
            plt.subplot(1,8,i-7)
            plt.imshow(x,cmap='gray')
            plt.axis('off')
        plt.savefig('/home/liuyang/save/base/%d/fm_superposition2.eps'%n,format='eps',dpi=1000)


# In[37]:

for i in range(1000):
    save_(i)
    save_1(i)
    save_2(i)
    save_3(i)
    save_4(i)

