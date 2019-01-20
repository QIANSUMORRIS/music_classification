#encoding:utf-8
import pandas as pd
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import sys
import time
 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# mpl.rcParams['font.sans-serif'] = [u'simHei']
# mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/result_process02.csv', sep =',')
# print(df.head(5))
df.dropna(axis = 0, how ='any', inplace = True) #删除数据中有空值的实例
# print(df.head(5))
# print(df.info())

x_train, x_test, y_train, y_test = train_test_split(df[['has_date','jieba_cut_content',\
                                                        'content_length_sema']],df['label'],\
                                                    test_size = 0.2, random_state = 0)

# print("训练集实例的个数是%d" % x_train.shape[0])
# print("测试集实例的个数是%d" % x_test.shape[0])
# print(x_train.head(10))
# print(x_test.head(10)) 
#================================================================================================
print('='*30 + '对分词后的邮件内容做tf-idf转化' + '='*30)
jieba_cut_content = list(x_train['jieba_cut_content'].astype('str'))
transformer = TfidfVectorizer(norm = 'l2', use_idf = True)#加载tf-idf模型
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)# fit_transform(jieba_cut_content)
# df1 = transformer.fit_transform(jieba_cut_content)
print('='*30 + '对tf-idf后的数值矩阵进行svd降维' + '='*30)
svd = TruncatedSVD(n_components=20)#降成20维
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)

print('='*30 + '合并处理后的矩阵' + '='*30)
data['has_date'] = list(x_train['has_date'])
data['content_length_sema'] = list(x_train['content_length_sema'])


print('='*30 + '朴素贝叶斯模型的加载及训练' + '='*30)  
nb = BernoulliNB(alpha = 1.0, binarize = 0.0005)
model = nb.fit(data, y_train)#训练模型

print('='*30 + '合并测试集数据矩阵' + '='*30)    
jieba_cut_content_test = list(x_test['jieba_cut_content'].astype('str'))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test['has_date'] = list(x_test['has_date'])
data_test['content_length_sema'] = list(x_test['content_length_sema'])

print('='*30 + '测试数据' + '='*30)
start = time.time()  
y_predict = model.predict(data_test)
end = time.time()
print('测试模型共消耗时间为：%0.2f'%(end-start))

print('='*30 + '评估模型召回率' + '='*30)   
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1mean = f1_score(y_test, y_predict)

print('='*30 + '打印预测结果如下' + '='*30)   
print('模型精确率为%0.5f' % precision)
print('模型召回率为%0.5f' % recall)
print('F1_mean为%0.5f' % f1mean)






