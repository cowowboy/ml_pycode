import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
# $1 = filename
# $2 = % = undersample amounts / tarining data
if __name__ == "__main__" :

    input_filename = sys.argv[1]   
    split_percent = round(float(sys.argv[2]),2)
    undersample_percent = round(float(sys.argv[3]),2)
    config_list = []
    
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

    # Adaboost
    print ("training......")
    sys.stdout.flush()
    while(undersample_percent < 1):
        config_list = []
        for i in range(1,26,2):
            for iteration in range(5,101,5):
                for num in range(2,6):
                    clf = AdaBoostClassifier(n_estimators=iteration ,base_estimator = DecisionTreeClassifier(max_depth = i),random_state = num)          
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
                    precis = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1score = 2 * precis * recall / ( precis + recall)
                    # find the best
                    if ( matrix[2,1:].sum() > 3 and matrix[0,2] < 98) or (matrix[1,1:].sum() > 4 and matrix[0,2] < 98) or (matrix[0,2] < 3 and matrix[3,1:].sum() == 1):
                        config_list.append(np.array_str(matrix))
                        config_list.append('randomstate : '+str(num)+' estimators : '+str(iteration)+' depth : '+str(i)+' precision : '+str(precis)+' recall : '+str(recall)+' class3 : '+str(matrix[2][2]))    
                    if ( f1score > 0.38 ):
                        config_list.append(np.array_str(matrix))
                        config_list.append('randomstate : '+str(num)+' estimators : '+str(iteration)+' depth : '+str(i)+' precision : '+str(precis)+' recall : '+str(recall)+' class3 : '+str(matrix[2][2]))
                        
        
        with open('adaboostconfig' + " %s" % undersample_percent +".txt",'w') as wfile:
            for str1 in config_list:
                wfile.write("%s\n" % str1)
        undersample_percent = undersample_percent + 0.10
