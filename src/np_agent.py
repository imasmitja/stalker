# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:44:12 2022

@author: Usuari
"""

#writting a numpy imnplementation of a Torch actor.



import numpy as np
import os
from configparser import ConfigParser


class np_rl_agent(object):
    def __init__(self):
        #w1(10x64)
        #w2(64x32)
        #w3(31x1)
        self.w1 = np.loadtxt('pretrained_values/w1.txt',delimiter=',').T
        self.w2 = np.loadtxt('pretrained_values/w2.txt',delimiter=',').T
        self.w3 = np.loadtxt('pretrained_values/w3.txt',delimiter=',').T
        #bias
        self.b1 = np.loadtxt('pretrained_values/b1.txt',delimiter=',').T
        self.b2 = np.loadtxt('pretrained_values/b2.txt',delimiter=',').T
        self.b3 = np.loadtxt('pretrained_values/b3.txt',delimiter=',').T
        
        
    #activation function
    def np_relu(self,x):
        for i in range(x.size):
            if x[i]<0.:
                x[i] = 0.
        return x
    
    def np_forward(self,x):
        #layer1 (input)
        prob1 = x.dot(self.w1) + self.b1
        prob1 = self.np_relu(prob1)
        
        #layer2 (hiden)
        prob2 = prob1.dot(self.w2) + self.b2
        prob2 = self.np_relu(prob2)
        
        #layer3 (output)
        mean = prob2.dot(self.w3) + self.b3
        
        return mean.clip(-1.0,1.0)
    
class np_rnn_rl_agent(object):
    def __init__(self):
        #values for agent: saca_lstm_l_v7_3_emofish
        #w1(10x64)
        #w2(64x32)
        #w3(31x1)
        self.w1rnn = np.loadtxt('pretrained_values/w1rnn.txt',delimiter=' ').T
        self.w2rnn = np.loadtxt('pretrained_values/w2rnn.txt',delimiter=' ').T
        self.w3rnn = np.loadtxt('pretrained_values/w3rnn.txt',delimiter=' ').T
        #bias
        self.b1rnn = np.loadtxt('pretrained_values/b1rnn.txt',delimiter=' ').T
        self.b2rnn = np.loadtxt('pretrained_values/b2rnn.txt',delimiter=' ').T
        self.b3rnn = np.loadtxt('pretrained_values/b3rnn.txt',delimiter=' ').T
        
        #LSTM parameters
        #ft
        self.whf = np.loadtxt('pretrained_values/whf.txt',delimiter=' ').T
        self.wif = np.loadtxt('pretrained_values/wif.txt',delimiter=' ').T
        self.bhf = np.loadtxt('pretrained_values/bhf.txt',delimiter=' ').T
        self.bif = np.loadtxt('pretrained_values/bif.txt',delimiter=' ').T
        #it
        self.whi = np.loadtxt('pretrained_values/whi.txt',delimiter=' ').T
        self.wii = np.loadtxt('pretrained_values/wii.txt',delimiter=' ').T
        self.bhi = np.loadtxt('pretrained_values/bhi.txt',delimiter=' ').T
        self.bii = np.loadtxt('pretrained_values/bii.txt',delimiter=' ').T
        #gt
        self.whg = np.loadtxt('pretrained_values/whg.txt',delimiter=' ').T
        self.wig = np.loadtxt('pretrained_values/wig.txt',delimiter=' ').T
        self.bhg = np.loadtxt('pretrained_values/bhg.txt',delimiter=' ').T
        self.big = np.loadtxt('big.txt',delimiter=' ').T
        #ot
        self.who = np.loadtxt('pretrained_values/who.txt',delimiter=' ').T
        self.wio = np.loadtxt('pretrained_values/wio.txt',delimiter=' ').T
        self.bho = np.loadtxt('pretrained_values/bho.txt',delimiter=' ').T
        self.bio = np.loadtxt('pretrained_values/bio.txt',delimiter=' ').T
        #Layer2
        #ft2
        self.whf2 = np.loadtxt('pretrained_values/whf2.txt',delimiter=' ').T
        self.wif2 = np.loadtxt('pretrained_values/wif2.txt',delimiter=' ').T
        self.bhf2 = np.loadtxt('pretrained_values/bhf2.txt',delimiter=' ').T
        self.bif2 = np.loadtxt('pretrained_values/bif2.txt',delimiter=' ').T
        #it2
        self.whi2 = np.loadtxt('pretrained_values/whi2.txt',delimiter=' ').T
        self.wii2 = np.loadtxt('pretrained_values/wii2.txt',delimiter=' ').T
        self.bhi2 = np.loadtxt('pretrained_values/bhi2.txt',delimiter=' ').T
        self.bii2 = np.loadtxt('pretrained_values/bii2.txt',delimiter=' ').T
        #gt2
        self.whg2 = np.loadtxt('pretrained_values/whg2.txt',delimiter=' ').T
        self.wig2 = np.loadtxt('pretrained_values/wig2.txt',delimiter=' ').T
        self.bhg2 = np.loadtxt('pretrained_values/bhg2.txt',delimiter=' ').T
        self.big2 = np.loadtxt('pretrained_values/big2.txt',delimiter=' ').T
        #ot2
        self.who2 = np.loadtxt('pretrained_values/who2.txt',delimiter=' ').T
        self.wio2 = np.loadtxt('pretrained_values/wio2.txt',delimiter=' ').T
        self.bho2 = np.loadtxt('pretrained_values/bho2.txt',delimiter=' ').T
        self.bio2 = np.loadtxt('pretrained_values/bio2.txt',delimiter=' ').T
        
        #LSTM Hidden states initial values
        self.ht0  = np.zeros(64) #Initial values for RNN
        self.ct0  = np.zeros(64) #Initial values for RNN
        self.ht02 = np.zeros(64) #Initial values for RNN
        self.ct02 = np.zeros(64) #Initial values for RNN

    #Sigmoid function
    def sigmoid(self,x):
        return 1. / (1. + np.exp(-x))
    #Tanh function
    def tanh(self,x):
        return np.tanh(x)
    #activation function
    def np_relu(self,x):
        for i in range(x.size):
            if x[i]<0.:
                x[i] = 0.
        return x
    
    def lstm1(self,x,h,c):
        #layer 1
        # import pdb; pdb.set_trace()
        ft = self.sigmoid(h.dot(self.whf)+self.bhf + x.dot(self.wif)+self.bif)
        it = self.sigmoid(h.dot(self.whi)+self.bhi + x.dot(self.wii)+self.bii)
        gt =    self.tanh(h.dot(self.whg)+self.bhg + x.dot(self.wig)+self.big)
        ot = self.sigmoid(h.dot(self.who)+self.bho + x.dot(self.wio)+self.bio)
        ct = np.multiply(ft,c) + np.multiply(it,gt)
        ht = np.multiply(self.tanh(ct),ot)
        return ht, ct
    
    def lstm2(self,x,h,c):
        #layer 2
        ft = self.sigmoid(h.dot(self.whf2)+self.bhf2 + x.dot(self.wif2)+self.bif2)
        it = self.sigmoid(h.dot(self.whi2)+self.bhi2 + x.dot(self.wii2)+self.bii2)
        gt =    self.tanh(h.dot(self.whg2)+self.bhg2 + x.dot(self.wig2)+self.big2)
        ot = self.sigmoid(h.dot(self.who2)+self.bho2 + x.dot(self.wio2)+self.bio2)
        ct = np.multiply(ft,c) + np.multiply(it,gt)
        ht = np.multiply(self.tanh(ct),ot)
        return ht, ct
        
    def np_forward(self,x):
        
        #layer1 (input)
        prob1 = x.dot(self.w1rnn) + self.b1rnn
        prob1 = self.np_relu(prob1)
        
        #LSTM (hidden)
        ht1,ct1 = self.lstm1(prob1, self.ht0, self.ct0 )
        ht2,ct2 = self.lstm2(ht1,self.ht02,self.ct02)
        
        #layer2 (hidden)
        prob2 = ht2.dot(self.w2rnn) + self.b2rnn
        prob2 = self.np_relu(prob2)
        
        #layer3 (output)
        mean = prob2.dot(self.w3rnn) + self.b3rnn
        
        #copy next values
        self.ht0  = ht1.copy()
        self.ct0  = ct1.copy()
        self.ht02 = ht2.copy()
        self.ct02 = ct2.copy() 
        
        return mean.clip(-1.0,1.0)    








