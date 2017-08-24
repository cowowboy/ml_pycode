import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
# $1 = filename
# $2 = % = undersample amounts / tarining data
if __name__ == "__main__" :

    input_filename = sys.argv[1]   
    split_percent = round(float(sys.argv[2]),2)
    undersample_percent = round(float(sys.argv[3]),2)
    drop_tablelist=[]
    print('Split :',split_percent,round(1-split_percent,2))
    print('undersample_percent : ',undersample_percent)
    # read csv data
    r_data = pd.read_csv(input_filename, dtype = 'object')

    # form datetime to timestamp
    datafirst = datetime.strptime("2015-01-01 00:00","%Y-%m-%d %H:%M")
    for i in range (0,round(r_data.shape[0])):    
        datetime_temp = datetime.strptime(r_data.loc[i]['time_1'].split("'")[1],"%Y-%m-%d %H:%M")
        r_data.loc[i]['time_1'] = (datetime_temp - datafirst).total_seconds() / 60
        
    # read columns values to list
    column_labels = list(r_data.columns.values)
    # remove time_1 column
    """column_labels.remove("time_1")"""
    # remove predict_status column, because they are answers
    column_labels.remove("predict_status")
    
    # iloc is index location
    # split data
    train_data = r_data[0 : round(r_data.shape[0] * split_percent)]
    test_data = r_data[round(r_data.shape[0] * split_percent):]
    
    # undersample for train_data
    for i in range(0,round(train_data.shape[0] * undersample_percent)):
        if train_data.iloc[i,-1] == '1.0':
            drop_tablelist.append(i)
            
    train_data = train_data.drop(train_data.index[drop_tablelist])

    # GaussianNB
    print ("training......")
    sys.stdout.flush()

    clf = GaussianNB()
           

    clf.fit(train_data.as_matrix([column_labels]) ,train_data.as_matrix(['predict_status']).reshape(-1))
    #print(clf.score(test_data.as_matrix([column_labels]),test_data.as_matrix(['predict_status']).reshape(-1)))

    
    y_pred = clf.predict(test_data.as_matrix([column_labels]))
    y_true = test_data.as_matrix(['predict_status'])
    matrix = confusion_matrix(y_true, y_pred)
    # compute tp , tn , fp , fn , precision , recall ,f1score
    tp = matrix[1:,1:].sum()
    tn = matrix[0,0]
    fn = matrix[1:,0].sum()
    fp = matrix[0,1:].sum()
    precis = round(tp / (tp + fp),3)
    recall = round(tp / (tp + fn),3)
    f1score = 2 * precis * recall / ( precis + recall)

    # find the best
    print (matrix)
    print ('TN : ',tn,' TP : ',tp,' FN : ',fn,' FP : ',fp)
    print ('precision : ',precis,'recall : ',recall)
    print ('f1score : ',f1score)

    