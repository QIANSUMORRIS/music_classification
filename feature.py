#coding:utf-8
import pandas as pd
import numpy as np
import glob#搜索歌单路径
from pydub.audio_segment import AudioSegment
from scipy.io import wavfile
from python_speech_features import mfcc
import os#.remove("")
import sys
import time#计算程序运行时间


def 获取歌单():#extract_label
    '''获取歌单，为正反向字典制作做准备'''
    data = pd.read_csv(歌单路径)
    data = data[['name','tag']]
    return data
 
def 获取单首歌曲特征(file):#fetch_index_label
    '''转换音乐文件格式并且提取其特征'''
    '''./data/music\\50 Cent - Ready For War.mp3'''
    items = file.split('.')
    file_format = items[-1].lower()#获取歌曲格式 mp3
    file_name = file[: -(len(file_format)+1)]#获取歌曲名称
    #把mp3格式的数据转化为wav
    if file_format != 'wav':
        '''把mp3格式转换为wav，保存至原文件夹中'''
        song = AudioSegment.from_file(file, format = 'mp3')
        file = file_name + '.wav'
        song.export(file, format = 'wav')
    try:
        '''提取wav格式歌曲特征'''
        rate, data = wavfile.read(file)
        mfcc_feas = mfcc(data, rate, numcep = 13, nfft = 2048)
        mm = np.transpose(mfcc_feas)
        mf = np.mean(mm ,axis = 1)# mf变成104维的向量
        mc = np.cov(mm)
        result = mf
        for i in range(mm.shape[0]):
            result = np.append(result, np.diag(mc, i))
#         os.remove(file)
        return result#返回1个104维的向量
    except Exception as msg:
        print(msg)
           
def 特征提取主函数():#主函数extract_and_export
    df = 获取歌单()
    name_label_list = np.array(df).tolist()
    name_label_dict = dict(map(lambda t: (t[0], t[1]), name_label_list))#歌单做成字典
    '''['回不去的过往','清新']'''
    '''{'回不去的过往':'清新',......}'''
    labels = set(name_label_dict.values())
    '''['清新  ','兴奋','快乐','','','',.....]'''
    #labelsIDS化
    label_index_dict = dict(zip(labels, np.arange(len(labels))))#歌曲标签数值映射
    '''{'清新  ':0,'兴奋':1,'快乐':2,'','','',.....}#一共有10个'''
#    print(label_index_dict)
#    for k in label_index_dict:
#        print(k)
#        print(label_index_dict[k])
#    sys.exit(0)

    all_music_files = glob.glob(歌曲源路径)
    '''./data/music\\50 Cent - Ready For War.mp3'''
    '''./data/music\\Am,Double D,Join - 她带我看过这世界.mp3'''
    all_music_files.sort()
    '''查找样本歌曲，获取样本'''
    loop_count = 0
    flag = True
    all_mfcc = np.array([])
    
    for file_name in all_music_files:
        '''获取样本所有歌曲的特征'''
        print('开始处理：' + file_name)#.replace('\xa0', '') .replace('\xa0', '')
        '''./data/music\\50 Cent - Ready For War.mp3'''
        music_name = file_name.split('\\')[-1].split('.')[-2].split('-')[-1]#\为转意字符
        music_name = music_name.strip()
        if music_name in name_label_dict:
            '''样本标签数值化'''
            label_index = label_index_dict[name_label_dict[music_name]]
            '''歌曲标签字典比对查询——重要！！！'''
            '''[0, 1, 3, 4, 2 ........]'''
            ff = 获取单首歌曲特征(file_name)#104维的向量
            ff = np.append(ff, label_index)#给特征值赋予标签 ，得到105维的向量
            '''给特征加标签，最后变为一个1行105维的向量'''
             
            if flag:
                all_mfcc = ff
                flag = False
            else:
                all_mfcc = np.vstack([all_mfcc, ff])
        else:
            print('无法处理：' + file_name.replace('\xa0', '') +'; 原因是： 列表中不存在此歌曲')
        print('looping-----%d' % loop_count)
        print('all_mfcc.shape:', end='')
        print(all_mfcc.shape)
        loop_count +=1 #i++ i++ i/= i*=
    #保存数据
    label_index_list = []
    for k in label_index_dict:
        #字典转为list再保存
        label_index_list.append([k, label_index_dict[k]])
    pd.DataFrame(label_index_list).to_csv(数值化标签路径, header = None, \
                                          index = False, encoding = 'utf-8')
    pd.DataFrame(all_mfcc).to_csv(歌曲特征文件存放路径, header= None, \
                                  index =False, encoding='utf-8')
    return all_mfcc
   






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







    
if __name__=='__main__':
    歌单路径 = './data/music_info.csv'#music_info_csv_file_path
    歌曲源路径 = './data/music/*.mp3'#music_audio_dir
    数值化标签路径 = './data/music_index_label.csv'#music_index_label_path
    歌曲特征文件存放路径 = './data/music_features.csv'#music_features_file_path
    start = time.time()
    特征提取主函数()
    end = time.time()
    print('总耗时%.2f秒'%(end - start))
        
       
    
    
    
    
    
    
    
    