# encoding:utf-8

def get(res,tes):
    #精确度
    n = len(res)
    truth = (res == tes)
    pre = 0
    for flag in truth:#[1, 0 , 0 ,1 ,1]
        if flag:
            pre += 1
    return (pre * 100) /n #百分比 ，返回的是res 和tes 之间的想等的概率
