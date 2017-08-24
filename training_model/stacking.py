import numpy as np
import os
import pandas as pd
import sys
import time
from datetime import datetime
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# $1 = filename
# $2 = split_percent
# $3 = % = undersample amounts / tarining data
if __name__ == "__main__" :

    input_filename = sys.argv[1]   
    split_percent = round(float(sys.argv[2]),2)
    undersample_percent = round(float(sys.argv[3]),2)
    drop_tablelist = []
    drop_tablelist_2 = []
    drop_tablelist_3 = []
    print('Split :',split_percent,round(1-split_percent,2))
    print('undersample_percent : ',undersample_percent)
    
    # read csv data
    r_data = pd.read_csv(input_filename, dtype = 'object')

    # form datetime to timestamp
    if ( input_filename.split("_")[-1].split(".")[0] == '2016'):
        datafirst = datetime.strptime("2016-01-01 00:00","%Y-%m-%d %H:%M")
        for i in range (0,round(r_data.shape[0])):    
            datetime_temp = datetime.strptime(r_data.loc[i]['time_1'].split("'")[1],"%Y-%m-%d %H:%M")
            r_data.loc[i]['time_1'] = (datetime_temp - datafirst).total_seconds() / 60
            datetime_temp = datetime.strptime(r_data.loc[i]['time_2'].split("'")[1],"%Y-%m-%d %H:%M")
            r_data.loc[i]['time_2'] = (datetime_temp - datafirst).total_seconds() / 60
    else:
        datafirst = datetime.strptime("2015-01-01 00:00","%Y-%m-%d %H:%M")
        for i in range (0,round(r_data.shape[0])):    
            datetime_temp = datetime.strptime(r_data.loc[i]['time_1'].split("'")[1],"%Y-%m-%d %H:%M")
            r_data.loc[i]['time_1'] = (datetime_temp - datafirst).total_seconds() / 60
            
    # read columns values to list
    column_labels = list(r_data.columns.values)
    # remove predict_status column, because they are answers
    column_labels.remove("predict_status")
    
    # iloc is index location
    # split data
    train_data = r_data[0 : round(r_data.shape[0] * split_percent)]
    train_data_2 = r_data[0 : round(r_data.shape[0] * split_percent)]
    train_data_3 = r_data[0 : round(r_data.shape[0] * split_percent)]
    test_data = r_data[round(r_data.shape[0] * split_percent):]
    
    # undersample for train_data
    for i in range(0,round(train_data.shape[0] * undersample_percent)):
        if train_data.iloc[i,-1] == '1.0':
            drop_tablelist.append(i)
    # customize undersample your train_data
    for i in range(0,round(train_data_2.shape[0] * 0.50)):
        if train_data_2.iloc[i,-1] == '1.0':
            drop_tablelist_2.append(i)
    for i in range(0,round(train_data_3.shape[0] * 0.60)):
        if train_data_3.iloc[i,-1] == '1.0':
            drop_tablelist_3.append(i)
    train_data = train_data.drop(train_data.index[drop_tablelist])
    train_data_2 = train_data_2.drop(train_data_2.index[drop_tablelist_2])
    train_data_3 = train_data_3.drop(train_data_3.index[drop_tablelist_3])
    
    # Stacking with 4 adaboost
    print ("training......")
    sys.stdout.flush()
    #clf2 = RandomForestClassifier(max_depth = 15 , random_state = 2,n_jobs = -1,n_estimators = 6)
    #clf3 = RandomForestClassifier(max_depth = 8 , random_state = 4,n_jobs = -1,n_estimators = 23)
    clf4 = AdaBoostClassifier(n_estimators=7 ,base_estimator = DecisionTreeClassifier(max_depth = 20),random_state = 5)
    #clf5 = AdaBoostClassifier(n_estimators=20 ,base_estimator = DecisionTreeClassifier(max_depth = 15),random_state = 2)
    clf7 = AdaBoostClassifier(n_estimators=10 ,base_estimator = DecisionTreeClassifier(max_depth = 5),random_state = 1)  
    clf6 = AdaBoostClassifier(n_estimators=25 ,base_estimator = DecisionTreeClassifier(max_depth = 13),random_state = 2)
    clf8 = AdaBoostClassifier(n_estimators=20 ,base_estimator = DecisionTreeClassifier(max_depth = 7),random_state = 5)
    clf9 = AdaBoostClassifier(n_estimators=5 ,base_estimator = DecisionTreeClassifier(max_depth = 19),random_state = 3)
    clf10 = AdaBoostClassifier(n_estimators=15 ,base_estimator = DecisionTreeClassifier(max_depth = 13),random_state = 4)
    #eclf = VotingClassifier(estimators=[('r2',clf2),('r4',clf4),('r5',clf5),('r6',clf6),('r7',clf7)],weights=[1,1,1,1,1],voting='hard',n_jobs=-1)
    #eclf.fit(train_data.as_matrix([column_labels]) ,train_data.as_matrix(['predict_status']).reshape(-1))
    
    # create model
    clf8.fit(train_data.as_matrix([column_labels]) ,train_data.as_matrix(['predict_status']).reshape(-1))
    #clf4.fit(train_data.as_matrix([column_labels]) ,train_data.as_matrix(['predict_status']).reshape(-1))
    clf7.fit(train_data.as_matrix([column_labels]) ,train_data.as_matrix(['predict_status']).reshape(-1))
    clf6.fit(train_data_2.as_matrix([column_labels]) ,train_data_2.as_matrix(['predict_status']).reshape(-1))
    clf9.fit(train_data_3.as_matrix([column_labels]) ,train_data_3.as_matrix(['predict_status']).reshape(-1))
    clf10.fit(train_data_3.as_matrix([column_labels]) ,train_data_3.as_matrix(['predict_status']).reshape(-1))
    # predit each model result and compare with real answers
    y_pred = clf8.predict(test_data.as_matrix([column_labels]))
    #y_pred_2 = clf4.predict(test_data.as_matrix([column_labels]))
    y_pred_3 = clf6.predict(test_data.as_matrix([column_labels]))
    y_pred_4 = clf7.predict(test_data.as_matrix([column_labels]))
    y_pred_5 = clf9.predict(test_data.as_matrix([column_labels]))
    y_pred_6 = clf10.predict(test_data.as_matrix([column_labels]))

    for ii,item in enumerate(y_pred_3):
        if item == "3.0" :
            y_pred[ii] = item
    for ii,item in enumerate(y_pred_4):
        if item == "3.0" :
            y_pred[ii] = item
    for ii,item in enumerate(y_pred_5):
        if item == "3.0" :
            y_pred[ii] = item
    for ii,item in enumerate(y_pred_6):
        if item == "2.0" :
            y_pred[ii] = item
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

    # print performance of stacking
    print (matrix)
    print ('TN : ',tn,' TP : ',tp,' FN : ',fn,' FP : ',fp)
    print ('precision : ',precis,'recall : ',recall)
    print ('f1score : ',f1score)

    
