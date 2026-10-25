import numpy as np
import random
import math

from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

    
def word2idx(word, words):
    if word in words.keys():
        return int(words[word])
    
    return 0

def pad_seq(dataset, max_len):
    output = []
    for seq in dataset:
        pad = np.zeros(max_len)
        pad[:len(seq)] = seq
        output.append(pad)
        
    return np.array(output)

def rna2vec(seqset):
    letters = ['A', 'C', 'G', 'U', 'N']

    words = np.array([letters[i] + letters[j] + letters[k]
                     for i in range(len(letters))
                     for j in range(len(letters))
                     for k in range(len(letters))])

    words = {word: i + 1 for i, word in enumerate(words)}

    outputs = []

    for seq in seqset:
        output = []

        converted_seq = dna2rna(seq)  # Assuming dna2rna function converts DNA to RNA

        for i in range(0, len(converted_seq) - 2):  # -2 so we can index 3 letters
            output.append(word2idx(converted_seq[i:i + 3], words))

        if sum(output) != 0:
            # pad individual sequence
            padded_seq = np.pad(output, (0, 275 - len(output)), 'constant', constant_values=0)
            outputs.append(padded_seq)

    return np.array(outputs)

def rna2vec_pretraining(seqset):
    letters = ['A','C','G','U','N']
    
    words = np.array([letters[i]+letters[j]+letters[k]
                     for i in range(len(letters)) 
                     for j in range(len(letters)) 
                     for k in range(len(letters))])
    words = np.unique(words) 
    # print("words:", len(words))
    words = {word:i+1 for i, word in enumerate(words)}
    
    ss = ['S','H','M','I','B','X','E']
    words_ss = np.array([i + j + k for i in ss for j in ss for k in ss])
    words_ss = np.unique(words_ss)
    # print("words_ss:", len(words_ss))
    words_ss = {word:i+1 for i, word in enumerate(words_ss)}
    
    outputs = []
    outputs_ss = []
    for seq, ss in seqset:
        output = [] 
        output_ss = []
        conveted_seq = dna2rna(seq)

        for i in range(0, len(conveted_seq)-1):
            output.append(word2idx(conveted_seq[i:i+3], words))
            output_ss.append(word2idx(ss[i:i+3], words_ss))

            if len(output) == 275:
                outputs.append(np.array(output))
                outputs_ss.append(np.array(output_ss))
                
                output = []
                output_ss = []
                
        # We'll handle the padding of individual sequences before appending to the outputs list
        if len(output) > 0:
            padded_output = np.pad(output, (0, 275 - len(output)), 'constant', constant_values=0)
            outputs.append(padded_output)

            padded_output_ss = np.pad(output_ss, (0, 275 - len(output_ss)), 'constant', constant_values=0)
            outputs_ss.append(padded_output_ss)

    return np.array(outputs), np.array(outputs_ss)

def seq2vec(seqset, max_len, n_vocabs, n_target_vocabs, words, words_ss):
    word_max_len= 3
    
    outputs = []
    outputs_ss = []
    for seq, ss in seqset:
        output = []
        output_ss = [] 
        i = 0
        while i < len(seq):
            flag=False
            for j in range(word_max_len, 0, -1):
                if i+j <=len(seq):
                    if word2idx(seq[i:i+j], words)!= 0:
                        flag = True
                        output.append(word2idx(seq[i:i+j], words))
                        output_ss.append(word2idx(ss[i:i+j], words_ss))

                        if len(output)==max_len:
                            outputs.append(np.array(output))
                            outputs_ss.append(np.array(output_ss))
                            output = [] 
                            output_ss = []
                        i+=j
                        break 
            if flag==False: 
                i+=1
        if len(output) != 0: 
            outputs.append(np.array(output))
            outputs_ss.append(np.array(output_ss))
        
    return pad_seq(outputs, max_len), pad_seq(outputs_ss, max_len)

def tokenize_sequences(seqset, max_len, n_vocabs, words, word_max_len=3):
    outputs = []
    for seq in seqset:
        output = []
        i = 0
        while i < len(seq): 
            flag=False
            for j in range(word_max_len, 0, -1): 
                if i+j <=len(seq): 
                    if word2idx(seq[i:i+j], words)!= 0: 
                        flag = True 
                        output.append(word2idx(seq[i:i+j], words)) 
                        if len(output)==max_len: 
                            outputs.append(np.array(output))
                            output = [] 
                        i+=j 
                        break 
            if flag==False:
                i+=1
                
        if len(output) != 0: 
            outputs.append(np.array(output))
        
    return pad_seq(outputs, max_len)

def str2bool(seq):
    out = []
    for s in seq:
        if s == "positive": 
            out.append(1)
        elif s == "negative": 
            out.append(0)
            
    return np.array(out)

def dna2rna(seq):
    mapping = {'A':'A','C':'C','G':'G', 'U':'U', 'T':'U'}
    result = ""
    for s in seq:
        if s in mapping.keys():
            result += mapping[s]
        else:
            result += 'N'
            
    return result
