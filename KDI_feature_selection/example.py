from __future__ import print_function
import sys
from KDI_feature_selection import DI
import scipy
import scipy.io as sio
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import matplotlib.pyplot as plt
import argparse
import os
from scipy.io import arff
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import functools


def load_csv(filepath):
    """
    读取包含基因名的csv文件
    --------------------
    :param filepath:
    :return: 基因数据，基因类别和基因名称
    """
    data = pd.read_csv(filepath, header=None, low_memory=False)
    da = np.array(data)
    gene = da[0, 0:len(da[0]) - 1]  # 获取基因名称
    data_set = da[1:, 0:len(da[0]) - 1]  # 获取基因数据
    # 获取基因标签
    label = da[1:, -1]
    return data_set, label, gene


def load_arff(filepath):
    """
    加载ARFF文件数据并进行处理
    -----------------------
    :param filepath: ARFF文件路径
    :return: 数据,类别和基因名
    """
    file_data, meta = arff.loadarff(filepath)
    x = []

    for row in range(len(file_data)):
        x.append(list(file_data[row]))

    df = pd.DataFrame(x)

    length = len(df.values[0, :])
    data = df.values[:, 0:length - 1]
    # 对标签进行处理
    label = df.values[:, -1]
    cla = []
    for i in label:
        test = int(i)
        cla.append(test)
    clas = np.array(cla)
    gg = meta.names()
    name = np.array(gg)
    gene = name[0:len(name) - 1]
    return data, clas, gene


def isNA(data):
    """
    判读是否有缺失值
    --------------
    :param data: 为二维numpy类型的数据
    :return: True——存在缺失值
    False——不存在缺失值
    """
    mat = pd.DataFrame(data)
    t = mat.isnull().any(axis=0)
    tt = np.array(t)
    if True in tt:
        return True
    else:
        return False

filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\clean1.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\Hill_Valley_with_noise_Training.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\ionosphere-10an-nn.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\Libras Movement.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\MC2.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\processed.cleveland.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\Sensorless_drive_diagnosis.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\sonar.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\spambase-10an-nn.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\wdbc-10an-nn.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\FCBF\lung-cancer.arff'
# filepath = r'C:\Users\Administrator\Desktop\FCBF\dataset\FCBF\Madelon.arff'

file, tmpfile = os.path.split(filepath)
filename, file_extens = os.path.splitext(tmpfile)

if file_extens == ".csv":
    X, Y, gene = load_csv(filepath)
if file_extens == '.arff':
    X, Y, gene = load_arff(filepath)

# 对缺失值进行填充
if isNA(X):
    print("存在缺失值，对缺失值进行处理")
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform")
    X_imputer = imputer.fit_transform(X)
else:
    print("没有缺失值")
    X_imputer = X

X_minmax = StandardScaler().fit_transform(X_imputer)

# 进行10折交叉验证
kfold_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# NBY分类器
nby = GaussianNB()
nby_mean = []

# KNN分类器
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2, metric='minkowski')
knn_mean = []

# SVM分类器
svm = SVC(C=1.0, kernel='linear', gamma='auto', random_state=42, decision_function_shape='ovr', probability=True)
svm_mean = []
'''高斯SVM'''
svm_rfb = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42, decision_function_shape='ovr', probability=False)
svm_rbf_mean = []

# logistic回归分类器
logistic = LogisticRegression(penalty='l2', C=1.0, random_state=42, multi_class='auto', dual=False)
logistic_mean = []

'''决策树分类器'''
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
dtc_mean = []
'''多层感知器分类器'''
mlp = MLPClassifier(random_state=42)
mlp_mean = []
# Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--option', type=str, default='SMK_CAN_187.mat')
# args = parser.parse_args()
#
# svm_tuned_params = [{'kernel': ['rbf'], 'gamma': [1e0,1e-1,1e-2,1e-3,1e-4], 'C': [1, 10, 100]}]
# svc = SVC(kernel='rbf')
    
# data = sio.loadmat(args.option)
# x_train = data['X']
# y_train = data['Y']
# print(x_train.shape, y_train.shape)
# y_train = y_train[:,0]
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)

epsilon = 1e-3
mu = 0.001
num_features = 100
type_Y = 'categorical'
rank_di = DI.di(X_minmax, Y, len(X_minmax[0]), type_Y, epsilon, mu, learning_rate = 0.1, iterations = 1000, verbose = True)

num = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

print("DI SCORES**********")
res = []
for i in range (20):
    selected_feats_di = np.argsort(rank_di)[:num[i]]
    x_train_selected_di = np.take(x_train, selected_feats_di, axis=1)

    score_di = 0
    for rs in range (5):
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
        clf = GridSearchCV(estimator=svc, param_grid=svm_tuned_params, cv=inner_cv)
        score_di += cross_val_score(clf, X=x_train_selected_di, y=y_train, cv=outer_cv)
    res.append(score_di.mean()/5)
    
plt.plot(num, res, marker='+')
plt.ylabel('KSVM Accuracy')
plt.xlabel('# features selected')
plt.grid(True)
plt.show()

print("full feature**********")
score_full = 0
for rs in range (5):
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=rs)
    clf = GridSearchCV(estimator=svc, param_grid=svm_tuned_params, cv=inner_cv)
    score_full += cross_val_score(clf, X=x_train, y=y_train, cv=outer_cv)
print('Full feature accuracy:',score_full.mean()/5)
