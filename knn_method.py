import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def main():
    args = sys.argv[1:]
    for input_file in args:
        myList = []

        with open(input_file,'r') as f1:
            for line in f1:
                myList.append([float(val) for val in line.split(',')])
        #X = np.loadtxt('train_file.txt', delimiter=",", unpack=False)
        X = np.array(myList)

        myLabels = []
        with open('labelsss.txt','r') as f2:
            for label in f2:
                myLabels.append(label.strip())

        y = np.array(myLabels)

        fold = 5
        skf = StratifiedKFold(n_splits=fold)
        f1 = 0.0
        for train_index, test_index in skf.split(X, y):
            #X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index, :], y[test_index,:]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #clf = SVC(kernel='linear')
            neigh = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree',leaf_size=30,metric='minkowski')
            neigh.fit(X_train, y_train)
            predicted = neigh.predict(X_test)
            f1 += f1_score(y_test, predicted, average='binary', pos_label='+')

        print f1 / fold

if __name__ == '__main__':
    main()



