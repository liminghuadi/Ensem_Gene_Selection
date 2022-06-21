# !/usr/bin/ python3
# -*- coding: UTF-8 -*-
#@author: xiaoshi
#@file: Original
#@Time: 2021/12/14 20:47
import time
import os
import warnings

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

seed = 42


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


def fill_nan(X):
    temp_data = X
    for i in range(len(temp_data)):
        for j in range(len(temp_data[0, :])):
            if np.isnan(temp_data[i, j]):
                temp_data[i, j] = 0

    return temp_data

def classificationResult(y_true, y_pred):
    # import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_score, recall_score
    # reportMat = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    recall = round((TP / float(TP + FN)), 4)
    precision = round((TP / float(TP + FP)), 4)
    # 计算正确率acc
    acc = round((TP + TN) / float(TP + TN + FP + FN), 4)
    # 计算f1-value值
    # f1 = round(f1_score(y_true, y_pred), 4)
    f1 = round((2 * precision * recall) / (precision + recall), 4)
    # precision = round((precision_score(y_true, y_pred)), 4)
    # recall = round((recall_score(y_true, y_pred)), 4)

    return acc, f1, precision, recall

def classification(clt, dataset, y, kflod):
    acc = cross_val_score(estimator=clt, X=dataset, y=y, cv=kflod, scoring='accuracy').mean()

    f1 = cross_val_score(clt, dataset, y, cv=kflod, scoring='f1_macro').mean()

    precision = cross_val_score(clt, dataset, y, cv=kflod, scoring='precision_macro').mean()
    recall = cross_val_score(clt, dataset, y, cv=kflod, scoring='recall_macro').mean()

    return acc, f1, precision, recall

# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\aml.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\arcenc_train.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Breast.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\breast1.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\CNS.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Colon.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\colonbreast.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\CrohnDisease.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\CrohnDisease1.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\CrohnDisease2.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\DLBCL.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\GCM.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Glioma.arff'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\HuntingtonDisease.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Leukemia.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Leukemia_3.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Leukemia_4.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Lung.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Lung_5.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Lymphoma.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Lymphoma_3.arff'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Madelon_train.arff'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\MLL.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Myeloma.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\N_A.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Ovarian.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Prostate.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Prostate.sboner.csv'
# filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\Prostate.singh.csv'
filepath = r'C:\Users\xiaoshi\Desktop\树模型\data\data\SRBCT.csv'


file, tmpfile = os.path.split(filepath)
filename, file_extens = os.path.splitext(tmpfile)

if file_extens == ".csv":
    X, Y, gene = load_csv(filepath)
else:
    X, Y, gene = load_arff(filepath)

# 对缺失值进行填充
if isNA(X):
    print("存在缺失值，对缺失值进行处理")
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform")
    X_imputer = imputer.fit_transform(X)
else:
    print("没有缺失值")
    X_imputer = X

X_minmax = MinMaxScaler().fit_transform(X_imputer)

# 十折交叉验证
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

'''支持向量机分类器'''
svm = SVC(C=1.0, kernel='rbf', random_state=seed, probability=True)
'''K近邻分类器'''
knn = KNeighborsClassifier()
'''高斯贝叶斯分类器'''
nb = GaussianNB()
'''随机森林分类器'''
rtf = RandomForestClassifier(random_state=seed)
'''决策树分类器'''
dt = DecisionTreeClassifier(random_state=seed)
'''logistic分类器'''
logistic = LogisticRegression(random_state=seed)
'''LDA分类器'''
lda = LinearDiscriminantAnalysis()
'''多层感知器分类器'''
mlp = MLPClassifier(random_state=seed)

svm_acc = []
svm_f1 = []
svm_precision = []
svm_recall = []

knn_acc = []
knn_f1 = []
knn_precision = []
knn_recall = []

nb_acc = []
nb_f1 = []
nb_precision = []
nb_recall = []

# rtf_acc = []
# rtf_f1 = []
# rtf_precision = []
# rtf_recall = []

dt_acc = []
dt_f1 = []
dt_precision = []
dt_recall = []

logistic_acc = []
logistic_f1 = []
logistic_precision = []
logistic_recall = []

lda_acc = []
lda_f1 = []
lda_precision = []
lda_recall = []

mlp_acc = []
mlp_f1 = []
mlp_precision = []
mlp_recall = []

print("文件:%s" % filename)

'''svm'''
acc_svm, f1_svm, precision_svm, recall_svm = classification(svm, X_minmax, Y, kflod=k_fold)
svm_acc.append(acc_svm)
svm_f1.append(f1_svm)
svm_precision.append(precision_svm)
svm_recall.append(recall_svm)
print("svm分类器:")
print("svm acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(svm_acc), np.mean(svm_f1), np.mean(svm_precision), np.mean(svm_recall)))

'''knn'''
acc_knn, f1_knn, precision_knn, recall_knn = classification(knn, X_minmax, Y, kflod=k_fold)
knn_acc.append(acc_knn)
knn_f1.append(f1_knn)
knn_precision.append(precision_knn)
knn_recall.append(recall_knn)
print("knn分类器:")
print("knn acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(knn_acc), np.mean(knn_f1), np.mean(knn_precision), np.mean(knn_recall)))

'''nb'''
acc_nb, f1_nb, precision_nb, recall_nb = classification(nb, X_minmax, Y, kflod=k_fold)
nb_acc.append(acc_nb)
nb_f1.append(f1_nb)
nb_precision.append(precision_nb)
nb_recall.append(recall_nb)
print("nb分类器:")
print("nb acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(nb_acc), np.mean(nb_f1), np.mean(nb_precision), np.mean(nb_recall)))

'''dt'''
acc_dt, f1_dt, precision_dt, recall_dt = classification(dt, X_minmax, Y, kflod=k_fold)
dt_acc.append(acc_dt)
dt_f1.append(f1_dt)
dt_precision.append(precision_dt)
dt_recall.append(recall_dt)
print("dt分类器:")
print("dt acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(dt_acc), np.mean(dt_f1), np.mean(dt_precision), np.mean(dt_recall)))

'''logistic'''
acc_logistic, f1_logistic, precision_logistic, recall_logistic = classification(logistic, X_minmax, Y, kflod=k_fold)
logistic_acc.append(acc_logistic)
logistic_f1.append(f1_logistic)
logistic_precision.append(precision_logistic)
logistic_recall.append(recall_logistic)
print("logistic分类器:")
print("logistic acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(logistic_acc), np.mean(logistic_f1), np.mean(logistic_precision), np.mean(logistic_recall)))

'''lda'''
acc_lda, f1_lda, precision_lda, recall_lda = classification(lda, X_minmax, Y, kflod=k_fold)
lda_acc.append(acc_lda)
lda_f1.append(f1_lda)
lda_precision.append(precision_lda)
lda_recall.append(recall_lda)
print("lda分类器:")
print("lda acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(lda_acc), np.mean(lda_f1), np.mean(lda_precision), np.mean(lda_recall)))

'''mlp'''
acc_mlp, f1_mlp, precision_mlp, recall_mlp = classification(mlp, X_minmax, Y, kflod=k_fold)
mlp_acc.append(acc_mlp)
mlp_f1.append(f1_mlp)
mlp_precision.append(precision_mlp)
mlp_recall.append(recall_mlp)
print("mlp分类器:")
print("mlp acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(mlp_acc), np.mean(mlp_f1), np.mean(mlp_precision), np.mean(mlp_recall)))

# print("文件:%s" % filename)
# print("svm分类器:")
# print("svm acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(svm_acc), np.mean(svm_f1), np.mean(svm_precision), np.mean(svm_recall)))
# print("knn分类器:")
# print("knn acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(knn_acc), np.mean(knn_f1), np.mean(knn_precision), np.mean(knn_recall)))
# print("nb分类器:")
# print("nb acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(nb_acc), np.mean(nb_f1), np.mean(nb_precision), np.mean(nb_recall)))
# # print("rtf分类器:")
# # print("rtf acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(rtf_acc), np.mean(rtf_f1), np.mean(rtf_precision), np.mean(rtf_recall)))
# print("dt分类器:")
# print("dt acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(dt_acc), np.mean(dt_f1), np.mean(dt_precision), np.mean(dt_recall)))
# print("logistic分类器:")
# print("logistic acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(logistic_acc), np.mean(logistic_f1), np.mean(logistic_precision), np.mean(logistic_recall)))
# print("lda分类器:")
# print("lda acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(lda_acc), np.mean(lda_f1), np.mean(lda_precision), np.mean(lda_recall)))
# print("mlp分类器:")
# print("mlp acc: %.4f, f1: %.4f, precision: %.4f, recall: %.4f" % (np.mean(mlp_acc), np.mean(mlp_f1), np.mean(mlp_precision), np.mean(mlp_recall)))

result_svm = [["Accuracy", np.mean(svm_acc)], ["F1-score",np.mean(svm_f1)], ["Precision",np.mean(svm_precision)], ["Recall",np.mean(svm_recall)]]
result_knn = [["Accuracy",np.mean(knn_acc)], ["F1-score",np.mean(knn_f1)], ["Precision",np.mean(knn_precision)], ["Recall",np.mean(knn_recall)]]
result_nb = [["Accuracy",np.mean(nb_acc)], ["F1-score",np.mean(nb_f1)], ["Precision",np.mean(nb_precision)], ["Recall",np.mean(nb_recall)]]
# result_rtf = [[np.mean(rtf_acc)], [np.mean(rtf_f1)], [np.mean(rtf_precision)], [np.mean(rtf_recall)]]
result_dt = [["Accuracy",np.mean(dt_acc)], ["F1-score",np.mean(dt_f1)], ["Precision",np.mean(dt_precision)], ["Recall",np.mean(dt_recall)]]
result_logistic = [["Accuracy",np.mean(logistic_acc)], ["F1-score",np.mean(logistic_f1)], ["Precision",np.mean(logistic_precision)], ["Recall",np.mean(logistic_recall)]]
result_lda = [["Accuracy",np.mean(lda_acc)], ["F1-score",np.mean(lda_f1)], ["Precision",np.mean(lda_precision)], ["Recall",np.mean(lda_recall)]]
result_mlp = [["Accuracy",np.mean(mlp_acc)], ["F1-score",np.mean(mlp_f1)], ["Precision",np.mean(mlp_precision)], ["Recall",np.mean(mlp_recall)]]


file_svm = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\svm.csv'
file_knn = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\knn.csv'
file_nb = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\nb.csv'
file_dt = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\dt.csv'
file_logistic = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\logistic.csv'
file_lda = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\lda.csv'
file_mlp = r'C:\Users\xiaoshi\Desktop\树模型\实验结果\mlp.csv'

pd.DataFrame(result_svm).to_csv(file_svm, index=False, mode='a+', header=False)
pd.DataFrame(result_knn).to_csv(file_knn, index=False, mode='a+', header=False)
pd.DataFrame(result_nb).to_csv(file_nb, index=False, mode='a+', header=False)
pd.DataFrame(result_dt).to_csv(file_dt, index=False, mode='a+', header=False)
pd.DataFrame(result_logistic).to_csv(file_logistic, index=False, mode='a+', header=False)
pd.DataFrame(result_lda).to_csv(file_lda, index=False, mode='a+', header=False)
pd.DataFrame(result_mlp).to_csv(file_mlp, index=False, mode='a+', header=False)