import pandas as pd
import numpy as np

'''
new_graph = pd.read_csv('new_graph.txt',sep='\t',header=None,dtype=str)
new_graph.columns = ['ID','TID','weight']
edges_data = new_graph[['ID','TID']]
'''
#print new_graph.shape #3,270,514 edges formed in the new graph

merged_data = pd.read_csv('mergedAgain.txt',sep='\t',header=None,dtype=str)
merged_data.columns = ['ID','TID','Content']

edges_data = pd.read_csv('complete_EL_nospace.txt',sep='\t',header=None,dtype=str)
edges_data.columns = ['ID','TID']

all_data = edges_data.join(merged_data['Content'])
edges = all_data.assign(edge_num=[0 + i for i in xrange(len(all_data))])[['edge_num'] + all_data.columns.tolist()]

content = edges[['edge_num','Content']]
#np.savetxt('Contents.txt', content.values, fmt='%s', delimiter="\t")

labels = pd.read_csv('new_graph_labels.txt',header=None,dtype=str)
lab = labels.assign(edge_num=[0 + i for i in xrange(len(labels))])[['edge_num'] + labels.columns.tolist()]
np.savetxt('AllLabels.txt', lab.values, fmt='%s', delimiter="\t")

'''
id_list_f = []
duplicates = np.unique(edges_data.values)
for p in duplicates:
    id_list_f.append(str(p).strip())
id_list = np.unique(id_list_f)
print len(id_list) #152642 nodes
'''


