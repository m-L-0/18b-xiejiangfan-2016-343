# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:29:42 2018

@author: Admin
"""
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

#导入数据
import_data2 = loadmat(r'./train/data2_train.mat')['data2_train']
import_data3 = loadmat(r'./train/data3_train.mat')['data3_train']
import_data5 = loadmat(r'./train/data5_train.mat')['data5_train']
import_data6 = loadmat(r'./train/data6_train.mat')['data6_train']
import_data8 = loadmat(r'./train/data8_train.mat')['data8_train']
import_data10 = loadmat(r'./train/data10_train.mat')['data10_train']
import_data11 = loadmat(r'./train/data11_train.mat')['data11_train']
import_data12 = loadmat(r'./train/data12_train.mat')['data12_train']
import_data14 = loadmat(r'./train/data14_train.mat')['data14_train']

#重构需要用到的类
new_datay_list = []
for i  in import_data2:
    c1 = list(i)
    c1.append(2)
    new_datay_list.append(c1)
for i  in import_data3:
    c1 = list(i)
    c1.append(3)
    new_datay_list.append(c1)
for i  in import_data5:
    c1 = list(i)
    c1.append(5)
    new_datay_list.append(c1)
for i  in import_data6:
    c1 = list(i)
    c1.append(6)
    new_datay_list.append(c1)
for i  in import_data8:
    c1 = list(i)
    c1.append(8)
    new_datay_list.append(c1)
for i  in import_data10:
    c1 = list(i)
    c1.append(10)
    new_datay_list.append(c1)
for i  in import_data11:
    c1 = list(i)
    c1.append(11)
    new_datay_list.append(c1)
for i  in import_data12:
    c1 = list(i)
    c1.append(12)
    new_datay_list.append(c1)
for i  in import_data14:
    c1 = list(i)
    c1.append(14)
    new_datay_list.append(c1)

new_datawithlabel_array = np.array(new_datay_list)


#标准化数据集并存储
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]



data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)

# 模型训练
clf = SVC(kernel='rbf',gamma='auto',C=9)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)
print(accuracy)  #正确率
joblib.dump(clf, "Spectral_Model.m")


##模型测试
testdata = loadmat(r'data_test_final.mat')['data_test_final']
test_data = preprocessing.StandardScaler().fit_transform(testdata)
clf = joblib.load('Spectral_Model.m')  #导入模型
predict_label = clf.predict(test_data) #模型预测
#print(predict_label)
#预测结果输出到testy.csv文件中
dfdata = pd.DataFrame(data = predict_label,columns=['y'])
datapath = 'testy.csv'
dfdata.to_csv(datapath, index = True)

