#coding:utf-8
'调参、训练并保存模型'
from sklearn import svm
import acc
import pandas as pd
from sklearn.utils import shuffle#洗牌
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib#保存模型模块 load,tf.saver.save()
#joblib
#tf.saver.save("/")
import sys
import time
#构建网格实验的参数及架构
def 网格交叉验证(X, Y):#internal_cross_validation
    parameters = {
        'kernel':('linear', 'rbf', 'poly'),
        'C':[0.1, 1],
        'probability':[True, False],
        'decision_function_shape':['ovo', 'ovr']
    }
    clf = GridSearchCV(svm.SVC(random_state = 0), param_grid = parameters, cv = 5)#固定格式
    print('开始交叉验证获取最优参数构建')
    clf.fit(X, Y)
    print('最优参数：', end = '')
    print(clf.best_params_)
    print('最优模型准确率：', end = '')
    print(clf.best_score_)
#开始交叉验证，返回最优参数    
def 交叉验证主函数(music_csv_file_path= None, data_percentage = 0.7): # cross_validation
    if not music_csv_file_path:
        music_csv_file_path = 歌曲特征文件存放路径
    print('开始读取数据：' + music_csv_file_path)
    data = pd.read_csv(music_csv_file_path, sep = ',', header = None, encoding = 'utf-8')
    sample_fact = 0.7
    if isinstance(data_percentage, float) and 0 < data_percentage < 1:
        sample_fact = data_percentage
    data = data.sample(frac = sample_fact).T
    X = data[:-1].T 
    Y = np.array(data[-1:])[0]
#    print(X)
#    print(Y)
#    sys.exit(0)
    网格交叉验证(X, Y)
#===========================================================================
#获得参数之后构建多项式模型
def 多项式模型(X, Y):
    """进行模型训练，并且计算训练集上预测值与label的准确性
    """
    clf = svm.SVC(kernel = 'rbf', C= 0.1, probability = True, decision_function_shape = 'ovo', random_state = 0)
    clf.fit(X, Y)
    res = clf.predict(X)
#     print(res)
#     sys.exit("53")
    restrain = acc.get(res,Y)
    return clf, restrain#返回模型及预测准确度
#开始训练模型
def 多次训练并保存模型(train_percentage = 0.7, fold = 5000, music_csv_file_path=None, model_out_f= None):#fit_dump_model
    """pass"""
    if not music_csv_file_path:
        music_csv_file_path = 歌曲特征文件存放路径
    data = pd.read_csv(music_csv_file_path, sep=',', header = None, encoding = 'utf-8')
    #trick
    max_train_source = None
    max_test_source = None
    max_source = None
    best_clf = None
    flag = True
    for index in range(1, int(fold)+1):#epoch 也可以写成1000
        print(index)
        shuffle_data = shuffle(data)#特征
        X = shuffle_data.T[:-1].T
        Y = np.array(shuffle_data.T[-1:])[0]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = train_percentage)#并未制定随机种子
        (clf, train_source) = 多项式模型(x_train, y_train)#返回的是模型及训练集上的准确率
        y_predict = clf.predict(x_test)
        test_source = acc.get(y_predict, y_test)#测试集的准确率
        source = 0.35 * train_source + 0.65 * test_source#模型综合准确率
        #记录最优模型
        #找出最大的精确率对应的模型：
        if flag:
            max_source = source
            max_train_source = train_source
            max_test_source = test_source
            best_clf = clf
            flag = False
        else:
            if max_source < source:
                max_source = source
                max_train_source = train_source
                max_test_source = test_source
                best_clf = clf
#模型的实时保存
#         if index % 10 == 0:
#             if not model_out_f:
#                 model_out_f = 模型保存路径
#                 joblib.dump(best_clf, model_out_f)
        print('第%d次epoch训练，训练集上的正确率为：%0.2f, 测试集上正确率为：%0.2f,加权平均正确率为：%0.2f'%(index , train_source,\
                                                                       test_source, source ))
    print('最优模型效果：训练集上的正确率为：%0.2f,测试集上的正确率为：%0.2f, 加权评均正确率为：%0.2f'%(max_train_source,\
                                                                     max_test_source, max_source))
    print('最优模型是：')
    print(best_clf) 
    #保存模型或模型持久化
    if not model_out_f:
        model_out_f = 模型保存路径
    joblib.dump(best_clf, model_out_f)
     


    
    
    
    

    
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    数值化标签路径 = './data/music_index_label.csv'#music_index_label_path
    歌曲特征文件存放路径 = './data/music_features.csv'#default_music_csv_file_path
    模型保存路径 = './data/music_model.pkl'#default_model_file_path
#第①步(运行完之后注释掉)
#     print('='*30 + '网格训练寻找最合适模型开始。。。' + '='*30)
#     start= time.time()
#     交叉验证主函数(music_csv_file_path= None, data_percentage = 0.7)
#     end = time.time()
#     print('寻找最佳模型共耗时%.2f'%(end-start))
#第②步
    print('='*30 + '网格训练寻找最合适模型开始。。。' + '='*30)
    start= time.time()
    多次训练并保存模型(music_csv_file_path=None, model_out_f= None)
    end = time.time()
    print('训练模型共耗时%.2f'%(end-start))
    
    
    
    
    
    