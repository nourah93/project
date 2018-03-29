import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
from gensim.models import doc2vec


merged_data = pd.read_csv('mergedAgain.txt',sep='\t',header=None,dtype=str)
merged_data.columns = ['ID','TID','Content']

doc2vec_data = merged_data['Content'].fillna(0)

'''
doc2vec_file = open("doc2vec_new.txt",'w')
for i in range(len(doc2vec_data)):
    regex = re.compile('(<xref.*</xref>)')
    if regex.search(str(doc2vec_data.iloc[i])):
        newstring = re.sub(regex, '', doc2vec_data.iloc[i])
        s = (newstring.translate(None, string.punctuation)).lower()
        f = ' '.join(s.split())
        word_list = f.split()
        filtered_words = [word for word in word_list if word not in stopwords.words('english')]
        f2 = ' '.join(filtered_words)
        doc2vec_file.write(f2 + '\n')
    else:
        doc2vec_file.write('0' + '\n')

doc2vec_file.close()
'''

'''
f = open('doc2vec_new.txt','r')
index = {}
i = 0
for line in f:
    index[i] = line.strip()
    i = i + 1
f.close()

lineedoc = []
keys = []
for key in index:
    if index[key] != '0':
        lineedoc.append(index[key])
        keys.append(key)


docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(lineedoc):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

model = doc2vec.Doc2Vec(docs, vector_size=128, window = 10, workers = 1)
#print len(model.docvecs) #84521

vec_dict = {}
for k, vec in zip(keys, model.docvecs):
    vec_dict[k] = vec

content_list = []
#out_file = open("doc2vecOutput.txt",'w')
with open('doc2vec_.txt','r') as f:
    for k in index:
        if k in vec_dict:
            content_list.append(np.array(vec_dict[k]))
            #out_file.write(str(vec_dict[k]))
        else:
            content_list.append(np.zeros(128))
            #out_file.write(str(np.zeros(128)))

#out_file.close()

content_dict = {}
d=0
for con in content_list:
    content_dict[d] = con
    d = d+1
'''

#file that concatenates unupdated node2vec and content
'''
t_file = open('train_file2.txt','w')
with open('graph_output.txt','r') as fn:
    next(fn)
    for line in fn:
        lines = line.strip().split(' ')
        key = lines[0]
        node_vec = (lines[1:129])
        content_vec = content_dict[int(key)]
        for vec in node_vec:
            t_file.write(vec + ',')
        for v in content_vec:
            t_file.write(str(v) + ',')
        t_file.write('\n')

t_file.close()
'''

#file that makes node2vec without content
'''
t_file = open('node2vec_corr.txt','w')
with open('graph_output_correct.txt','r') as fn:
    next(fn)
    for line in fn:
        lines = line.strip().split(' ')
        key = lines[0]
        node_vec = (lines[1:129])
        for vec in node_vec:
            t_file.write(vec + ',')
        t_file.write('\n')

t_file.close()
'''




'''
#updating the weights
u_file = open('updated_graph_tunedoc.txt','w')
with open('new_graph.txt','r') as file:
    for line in file:
        lines = line.strip().split('\t')
        vec1 = content_dict[int(lines[0])]
        vec2 = content_dict[int(lines[1])]
        weight = lines[2]

        if np.all(vec1==0) or np.all(vec2==0):
            u_file.write(lines[0] + '\t' + lines[1] + '\t' + weight + '\n')
        else:
            sim = cosine_similarity(vec1.reshape(1,-1),vec2.reshape(1,-1))
            updated_weight = sim * int(weight)
            u_file.write(lines[0] + '\t' + lines[1] + '\t' + str(updated_weight[0][0]) + '\n')

u_file.close()
'''
