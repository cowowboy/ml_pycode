import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# $1 = filename
# $2 = % = undersample amounts / tarining data
if __name__ == "__main__" :

    input_filename = sys.argv[1]   
    split_percent = round(float(sys.argv[2]),2)
    undersample_percent = round(float(sys.argv[3]),2)
    
    config_list= []
    drop_tablelist = []
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

    # randomforest config
    print ("training......")
    sys.stdout.flush()
    for iteration in range(2,101,5):
        for ii in range(1,30,2):
            for num in (6,10):
                clf = RandomForestClassifier(max_depth = ii , random_state = num,n_jobs = -1,n_estimators = iteration)

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
                if ( matrix[2][2] > 2):
                    config_list.append('seed: '+str(num)+'estimators: '+str(iteration)+' depth: '+str(ii)+' TN: '+str(tn)+' TP :'+str(tp)+' FN :'+str(fn)+' FP :'+str(fp)+' precis :'+str(precis)+' recall :'+str(recall)+'class3 :'+str(matrix[2][2]))
                    print(config_list)
                    """print ('estimators : ',iteration)
                    print ('depth : ',ii)
                    print (matrix)
                    print ('TN : ',tn,' TP : ',tp,' FN : ',fn,' FP : ',fp)
                    print ('precision : ',precis,'recall : ',recall)
                    print ('f1score : ',f1score)"""
                if (  f1score > 0.38 ):
                    config_list.append('seed: '+str(num)+'estimators: '+str(iteration)+' depth: '+str(ii)+' TN:'+str(tn)+' TP :'+str(tp)+' FN :'+str(fn)+' FP :'+str(fp)+' precis :'+str(precis)+' recall :'+str(recall)+'class3 :'+str(matrix[2][2]))
                    print(config_list)

    with open("randomconfig.txt",'w') as wfile:
        for item in config_list:
            wfile.write("%s\n" %item)