import pandas as pd
from sklearn import tree
from itertools import izip
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#edges_data = pd.read_csv('complete_EL_nospace.txt',sep='\t',header=None,dtype=str)
#edges_data.columns = ['ID','TID']

#to get labels
'''
link_data = pd.read_csv('Link_Info.tsv',sep='\t',header=None,dtype=str)
link_data.columns = ['ID','TID','Label']

dict_links = link_data.set_index(['ID','TID'])['Label'].to_dict()
label_file = open('new_graph_labels.txt','w')
with open('complete_EL_nospace.txt','r') as f:
    for lines in f:
        line = lines.strip().split('\t')
        label_file.write(dict_links[line[0],line[1]])
        label_file.write('\n')
label_file.close()
'''

#dictionary has idofedge -> label
'''
labels_dict = {}
with open('AllLabels.txt','r') as f1:
    for line in f1:
        lines = line.strip().split('\t')
        labels_dict[lines[0]] = lines[1]
'''


#get the file we need to train with its labels
'''
x_file = open('to_train_doc.txt','w')
y_file = open('labelsss.txt','w')
with open("up_graph_output_tunedoc.txt" ,'r') as f:
    next(f)
    for lines in f:
        line = lines.strip().split(' ')
        key = line[0]
        values = (line[1:129])
        for v in values:
            x_file.write(v + ',')
        y_file.write(labels_dict[key] + '\n')
        x_file.write('\n')

x_file.close()
'''

#get file for node2vec+content with labels for matlab
'''
t_file = open('con_train_labels.txt','w')
with open("con_train_file.txt") as textfile1, open("labelsss.txt") as textfile2:
    for x, y in zip(textfile1, textfile2):
        t_file.write(x.strip() + ',' + y)
t_file.close()
'''
'''
def cutoff_predict(clf,X,cutoff):
    return (clf.predict_proba(X)[:,1]>cutoff).astype(int)

def custom_f1(cutoff):
    def f1_cutoff(clf,X,y):
        ypred = cutoff_predict(clf,X,cutoff)
        return f1_score(y,ypred)

    return f1_cutoff
    '''


#def main():
    #args = sys.argv[1:]
    #for input_file in args:
myList = []

with open('trainn_no_con.txt','r') as f1:
    for line in f1:
        myList.append([float(val) for val in line.split(',')])
#X = np.loadtxt('train_file.txt', delimiter=",", unpack=False)
X = np.array(myList)

myLabels = []
with open('labelsss.txt','r') as f2:
    for label in f2:
        myLabels.append(label.strip())

y = np.array(myLabels)
'''
for index,item in enumerate(y):
    if item == '-':
        y[index] = 1
    else:
        y[index] = 0
        '''


fold = 5
skf = StratifiedKFold(n_splits=fold)
f1 = 0.0
recall =0.0
precision = 0.0
for train_index, test_index in skf.split(X, y):
    #X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index, :], y[test_index,:]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #clf = tree.DecisionTreeClassifier()
    #clf = MLPClassifier(solver='lbfgs', alpha=0.001,hidden_layer_sizes = (100), activation='logistic')
    logreg = LogisticRegression(penalty='l2', C=10,random_state=0,class_weight='balanced')
    logreg.fit(X_train, y_train)
    predicted = logreg.predict(X_test)
    f1 += f1_score(y_test, predicted, average='binary', pos_label='+')
    recall += recall_score(y_test,predicted,average='binary', pos_label='+')
    precision += precision_score(y_test, predicted, average='binary', pos_label='+')

print ("precision= {}, recall = {}, f1 = {}".format(precision / fold, recall / fold, f1 / fold))

'''
#scores = []
#logreg = LogisticRegression(penalty='l2', C=10,random_state=0,class_weight='balanced')
#validated = cross_val_score(logreg,X,y,cv=10,scoring= custom_f1(0.1))
#scores.append(validated)
'''
#if __name__ == '__main__':
 #   main()
