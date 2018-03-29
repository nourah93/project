import pandas as pd
import numpy as np
from itertools import permutations


merged_data = pd.read_csv('mergedAgain.txt',sep='\t',header=None,dtype=str)
merged_data.columns = ['ID','TID','Content']

edges_data = pd.read_csv('complete_EL_nospace.txt',sep='\t',header=None,dtype=str)
edges_data.columns = ['ID','TID']

link_data = pd.read_csv('Link_Info.tsv',sep='\t',header=None,dtype=str)
link_data.columns = ['ID','TID','Label']


all_data = edges_data.join(merged_data['Content'])
edges = all_data.assign(edge_num=[0 + i for i in xrange(len(all_data))])[['edge_num'] + all_data.columns.tolist()]


'''
sample_edges = edges_data.head(15)
sample_data = sample_edges.join(merged_data['Content'].head(15))
edges = sample_data.assign(edge_num=[0 + i for i in xrange(len(sample_data))])[['edge_num'] + sample_data.columns.tolist()]
'''
#check number of nodes in graph
'''
id_list_f = []
duplicates = np.unique(edges_data.values)
for p in duplicates:
    id_list_f.append(str(p).strip())
id_list = np.unique(id_list_f)
#print len(id_list) #76557
'''
'''
edges_data = pd.read_csv('try_graph.txt',sep='\t',header=None,dtype=str)
edges_data.columns = ['edge_num','ID','TID']
'''

group1 = edges.groupby('TID')['edge_num'].unique()
less_groups1 = group1[group1.apply(lambda x: len(x)>1)]
self_group1 = group1[group1.apply(lambda x: len(x)==1)]

group2 = edges.groupby('ID')['edge_num'].unique()
less_groups2 = group2[group2.apply(lambda x: len(x)>1)]
self_group2 = group2[group2.apply(lambda x: len(x)==1)]


my_list1 = []
my_list2 = []
my_list3 = []

for j in less_groups1.values:
    my_list1.append(j)


for k in less_groups2.values:
    my_list2.append(k)

self_group = self_group1.append(self_group2)

for p in self_group.values:
    my_list3.append(p[0])



tmp = set()
e_file = open('new_graph_correct.txt','w')

for lst1 in my_list1:
    for i in permutations(lst1.tolist(), 2):
        pair1 = " ".join(str(sorted(i)))
        if pair1 not in tmp:
            tmp.add(pair1)
            e_file.write(str(i[0]) + '\t' + str(i[1]) + '\t' + '2\n')


for lst2 in my_list2:
    for j in permutations(lst2.tolist(),2):
        pair2 = " ".join(str(sorted(j)))
        if pair2 not in tmp:
            tmp.add(pair2)
            e_file.write(str(j[0]) + '\t' + str(j[1]) + '\t' + '1\n')

for lst3 in my_list3:
    e_file.write(str(lst3) + '\t' + str(lst3) + '\t' + '1\n')

e_file.close()


