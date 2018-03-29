import pandas as pd
import numpy as np
from collections import Counter
import itertools

def indices( mylist, value):
    return [i for i,x in enumerate(mylist) if x==value]

def get_sample(targets):
    all_samples = []
    count = Counter(targets)
    unique_t = np.unique(targets)
    target_list = targets.tolist()
    for t in unique_t:
        count[t] = count[t] * 0.1
        if count[t] > 4:
            valid_samples = [index for index, value in enumerate(target_list) if value == t]
            #for tr in target_list:
                #if t == tr:

            #valid_samples = [indices(target_list,tr) for tr in target_list if tr==t]
            samples = np.random.choice(valid_samples,int(count[t]),replace=False)
            for s in samples:
                all_samples.append(s)

    return all_samples



edges_data = pd.read_csv('complete_EL_nospace.txt',sep='\t',header=None,dtype=str)
edges_data.columns = ['ID','TID']
edges = edges_data.assign(edge_num=[0 + i for i in xrange(len(edges_data))])[['edge_num'] + edges_data.columns.tolist()]

#np.savetxt('id_edge.txt', edges.values, fmt='%s', delimiter="\t")

sources = edges['ID'].values
targets = edges['TID'].values

index_test = get_sample(targets)
label_dict = {}
with open('AllLabels.txt','r') as f:
    for lines in f:
        line = lines.strip().split('\t')
        label_dict[line[0]]=line[1]

edges_dict = {}
i=0
with open('complete_EL_nospace.txt','r') as f2:
    for lines in f2:
        edges_dict[i] = lines.strip()
        i = i+1

'''
for ind in index_test:
    print edges_dict[ind] + ' ' + label_dict[str(ind)] + '\n'
'''

vec_dict = {}
with open('graph_output.txt','r') as f3:
    next(f3)
    for line in f3:
        lines = line.strip().split(' ')
        key = lines[0]
        values = (lines[1:129])
        new_values = [float(i) for i in values]
        vec_dict[key] = new_values

print vec_dict['126345']

'''
all_index = np.array(range(0,152663))
index_train =  (np.setdiff1d(all_index, index_test))

y_test = [label_dict[str(ind)] for ind in index_test]
y_train = [label_dict[str(ind)] for ind in index_train]

x_test = [vec_dict[str(ind)] for ind in index_test]
x_train = [vec_dict[str(ind)] for ind in index_train]

print len(x_test)
print len(x_train)
print len(y_test)
print len(y_train)
'''










