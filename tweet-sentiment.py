#!/usr/bin/env python
# coding: utf-8

# # 数据挖掘

# ## 姓名：刘思雯  学号：3120220948

# ## 1.数据集 Tweet Sentiment's Impact on Stock Returns

# In[1]:


# 导入必要的包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from math import isnan
import math


# ### 1.1查看数据集

# In[3]:


#查看数据集下的数据文件
import os
for dirname, _, filenames in os.walk('.\data\Tweet Sentiment Impact on Stock Returns'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#数据集文件包含full_dataset和reduced_dataset两个数据集
#这里以reduced_dataset数据集为例进行研究


# ### 1.2 读取数据集

# In[14]:


#读取数据集
path = './data/Tweet Sentiment Impact on Stock Returns/reduced_dataset-release.csv'
data = pd.read_csv(path,index_col=0,low_memory=False)
data.head()# 默认展示前五行数据


# In[7]:


data.dtypes # 每列数据的数据类型


# In[24]:


#处理数据-修改有误的数据类型
data["LSTM_POLARITY"] = pd.to_numeric(data["LSTM_POLARITY"],errors='coerce')
data["TEXTBLOB_POLARITY"] = pd.to_numeric(data["TEXTBLOB_POLARITY"],errors='coerce')
data.dtypes # 每列数据的数据类型


# In[8]:


data.shape # 数据集的大小


# ## 2.数据分析要求

# ### 2.1数据摘要

# 数据集各属性含义：
# * TWEET: Text of the tweet. (String)
# * STOCK: Company's stock mentioned in the tweet. (String)
# * DATE: Date the tweet was posted. (Date)
# * LAST_PRICE: Company's last price at the time of tweeting. (Float)
# * 1_DAY_RETURN: Amount the stock returned or lost over the course of the next day after being tweeted about. (Float)
# * 2_DAY_RETURN: Amount the stock returned or lost over the course of the two days after being tweeted about. (Float)
# * 3_DAY_RETURN: Amount the stock returned or lost over the course of the three days after being tweeted about. (Float)
# * 7_DAY_RETURN: Amount the stock returned or lost over the course of the seven days after being tweeted about. (Float)
# * PX_VOLUME: Volume traded at the time of tweeting. (Integer)
# * VOLATILITY_10D: Volatility measure across 10 day window. (Float)
# * VOLATILITY_30D: Volatility measure across 30 day window. (Float)
# * LSTM_POLARITY: Labeled sentiment from LSTM. (Float)
# * TEXTBLOB_POLARITY: Labeled sentiment from TextBlob. (Float)
# * MENTION: Number of times the stock was mentioned in the tweet. (Integer)

# 2.1.1 标称属性

# In[15]:


# 由上面对数据集分析可知，该数据集的标称属性有"TWEET、"STOCK"、"DATE"、"MENTION"四个标称属性
# 下面给出每个属性取值的频数
#（1）TWEET
pd.value_counts(data['TWEET'])


# In[16]:


#（2）STOCK
pd.value_counts(data['STOCK'])


# In[17]:


#（3）DATE
pd.value_counts(data['DATE'])


# In[18]:


#（4）MENTION
pd.value_counts(data['MENTION'])


# 2.1.2 数值属性

# In[19]:


# 这里的数值属性包括 LAST_PRICE、1_DAY_RETURN、2_DAY_RETURN、3_DAY_RETURN、7_DAY_RETURN、PX_VOLUME,
#VOLATILITY_10D、VOLATILITY_30D、LSTM_POLARITY和 TEXTBLOB_POLARITY
# 对数值属性的 5 数进行概括
digital_data = ['LAST_PRICE','1_DAY_RETURN','2_DAY_RETURN','3_DAY_RETURN','7_DAY_RETURN','PX_VOLUME','VOLATILITY_10D','VOLATILITY_30D','LSTM_POLARITY','TEXTBLOB_POLARITY']
data[digital_data].describe()


# In[20]:


#给出各数值属性缺失值个数
print("'LAST_PRICE'：",data['LAST_PRICE'].isnull().sum())
print("'1_DAY_RETURN'：",data['1_DAY_RETURN'].isnull().sum())
print("'2_DAY_RETURN'：",data['2_DAY_RETURN'].isnull().sum())
print("'3_DAY_RETURN'：",data['3_DAY_RETURN'].isnull().sum())
print("'7_DAY_RETURN'：",data['7_DAY_RETURN'].isnull().sum())
print("'PX_VOLUME'：",data['PX_VOLUME'].isnull().sum())
print("'VOLATILITY_10D'：",data['VOLATILITY_10D'].isnull().sum())
print("'VOLATILITY_30D'：",data['VOLATILITY_30D'].isnull().sum())
print("'LSTM_POLARITY'：",data['LSTM_POLARITY'].isnull().sum())
print("'TEXTBLOB_POLARITY'：",data['TEXTBLOB_POLARITY'].isnull().sum())


# 由此可见，各数值属性存在大量的缺失值

# ### 2.2数据可视化

# （1）绘制LAST_PRICE的直方图、盒图、qq图

# In[9]:


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("LAST_PRICE hist")
data['LAST_PRICE'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("LAST_PRICE box")
data['LAST_PRICE'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['LAST_PRICE'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['LAST_PRICE'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['LAST_PRICE'], dist="norm", plot=plt)
plt.show()


# （2）绘制1_DAY_RETURN的直方图、盒图、qq图

# In[24]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("1_DAY_RETURN hist")
data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("1_DAY_RETURN box")
data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['1_DAY_RETURN'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['1_DAY_RETURN'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['1_DAY_RETURN'], dist="norm", plot=plt)
plt.show()


# （3）绘制2_DAY_RETURN的直方图、盒图、qq图

# In[25]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("2_DAY_RETURN hist")
data['2_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("2_DAY_RETURN box")
data['2_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['2_DAY_RETURN'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['2_DAY_RETURN'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['2_DAY_RETURN'], dist="norm", plot=plt)
plt.show()


# （4）绘制3_DAY_RETURN的直方图、盒图、qq图

# In[26]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("3_DAY_RETURN hist")
data['3_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("3_DAY_RETURN box")
data['3_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['3_DAY_RETURN'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['3_DAY_RETURN'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['3_DAY_RETURN'], dist="norm", plot=plt)
plt.show()


# （5）绘制7_DAY_RETURN的直方图、盒图、qq图

# In[27]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("7_DAY_RETURN hist")
data['7_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("7_DAY_RETURN box")
data['7_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['7_DAY_RETURN'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['7_DAY_RETURN'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['7_DAY_RETURN'], dist="norm", plot=plt)
plt.show()


# （6）绘制PX_VOLUME的直方图、盒图、qq图

# In[28]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("PX_VOLUME hist")
data['PX_VOLUME'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("PX_VOLUME box")
data['PX_VOLUME'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['PX_VOLUME'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['PX_VOLUME'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['PX_VOLUME'], dist="norm", plot=plt)
plt.show()


# （7）绘制VOLATILITY_10D的直方图、盒图、qq图

# In[29]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("VOLATILITY_10D hist")
data['VOLATILITY_10D'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("VOLATILITY_10D box")
data['VOLATILITY_10D'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['VOLATILITY_10D'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['VOLATILITY_10D'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['VOLATILITY_10D'], dist="norm", plot=plt)
plt.show()


# （8）绘制VOLATILITY_30D的直方图、盒图、qq图

# In[31]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("VOLATILITY_30D hist")
data['VOLATILITY_30D'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("VOLATILITY_30D box")
data['VOLATILITY_30D'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['VOLATILITY_30D'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['VOLATILITY_30D'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['VOLATILITY_30D'], dist="norm", plot=plt)
plt.show()


# （9）绘制LSTM_POLARITY的直方图、盒图、qq图

# In[32]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("LSTM_POLARITY hist")
data['LSTM_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("LSTM_POLARITY box")
data['LSTM_POLARITY'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['LSTM_POLARITY'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['LSTM_POLARITY'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['LSTM_POLARITY'], dist="norm", plot=plt)
plt.show()


# （10）绘制TEXTBLOB_POLARITY的直方图、盒图、qq图

# In[33]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("TEXTBLOB_POLARITY hist")
data['TEXTBLOB_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("TEXTBLOB_POLARITY box")
data['TEXTBLOB_POLARITY'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['TEXTBLOB_POLARITY'],dist="norm",plot=plt)
# 去掉缺失值绘制 q-q 图
plt.subplot(2,2,4)
data_drop=pd.DataFrame(data['TEXTBLOB_POLARITY'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['TEXTBLOB_POLARITY'], dist="norm", plot=plt)
plt.show()


# ## 3.数据缺失值处理

# 统计缺失值

# In[5]:


#统计缺失值
def missing_data(datatodel):
    missing_num = datatodel.isnull().sum()
    missing_percent = missing_num/datatodel.shape[0]*100
    concat_data = pd.concat([missing_num,missing_percent],axis=1,keys=['missing_num','missing_percent'])
    concat_data['Types'] = datatodel.dtypes
    return concat_data
missing_data(data)


# 除了TWEET,其他属性的数据值均存在较多的缺失，可能是无法获取
# 接下来的缺失值处理以1_DAY_RETURN和LSTM_POLARITY为例

# ### 3.1剔除缺失值

# In[36]:


del_null_data = data.copy(deep=True)
del_null_data = del_null_data.dropna()
# 处理缺失数据后的数据展示
missing_data(del_null_data)


# In[37]:


# 1_DAY_RETURN 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("1_DAY_RETURN hist")
data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new 1_DAY_RETURN hist")
del_null_data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("1_DAY_RETURN box")
data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new 1_DAY_RETURN box")
del_null_data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['1_DAY_RETURN'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(del_null_data['1_DAY_RETURN'],dist="norm",plot=plt)
plt.show()


# In[38]:


# LSTM_POLARITY可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("LSTM_POLARITY hist")
data['LSTM_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new LSTM_POLARITY hist")
del_null_data['LSTM_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("LSTM_POLARITY box")
data['LSTM_POLARITY'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new LSTM_POLARITY box")
del_null_data['LSTM_POLARITY'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['LSTM_POLARITY'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(del_null_data['LSTM_POLARITY'],dist="norm",plot=plt)
plt.show()


# In[40]:


del_null_data['1_DAY_RETURN'].describe() # 缺失部分剔除后数据的 5 数概况


# In[41]:


del_null_data['LSTM_POLARITY'].describe() # 缺失部分剔除后数据的 5 数概况


# ### 3.2用最高频率填补缺失值

# In[25]:


# 用最高频率来填补缺失值--此处使用深拷贝，否则会改变原值
fill_data_with_most_frequency = data.copy(deep=True)
# 对'1_DAY_RETURN' 进行最高频率值填补缺失值
word_counts = Counter(fill_data_with_most_frequency['1_DAY_RETURN'])
top = word_counts.most_common(1)[0][0]
fill_data_with_most_frequency['1_DAY_RETURN'] = fill_data_with_most_frequency['1_DAY_RETURN'].fillna(top)
# 对'LSTM_POLARITY' 进行最高频率值填补缺失值
word_counts = Counter(fill_data_with_most_frequency['LSTM_POLARITY'])
top = word_counts.most_common(1)[0][0]
fill_data_with_most_frequency['LSTM_POLARITY'] = fill_data_with_most_frequency['LSTM_POLARITY'].fillna(top)
# 查看填充后是否还有数据缺失
missing_data(fill_data_with_most_frequency) 


# In[10]:


# 1_DAY_RETURN 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("1_DAY_RETURN hist")
data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new 1_DAY_RETURN hist")
fill_data_with_most_frequency['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("1_DAY_RETURN box")
data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new 1_DAY_RETURN box")
fill_data_with_most_frequency['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['1_DAY_RETURN'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(fill_data_with_most_frequency['1_DAY_RETURN'],dist="norm",plot=plt)
plt.show()


# In[26]:


# LSTM_POLARITY可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("LSTM_POLARITY hist")
data['LSTM_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new LSTM_POLARITY hist")
fill_data_with_most_frequency['LSTM_POLARITY'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("LSTM_POLARITY box")
data['LSTM_POLARITY'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new LSTM_POLARITY box")
fill_data_with_most_frequency['LSTM_POLARITY'].plot(kind='box',notch=True,grid=True)
# q-q 图
plt.subplot(3,2,5)
stats.probplot(data['LSTM_POLARITY'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(fill_data_with_most_frequency['LSTM_POLARITY'],dist="norm",plot=plt)
plt.show()


# In[27]:


# 对填充后的新数据进行描述
fill_data_with_most_frequency[['1_DAY_RETURN','LSTM_POLARITY']].describe()


# ### 3.3 通过属性的相关关系来填补缺失值

# In[28]:


# 查看相关的属性关系
data.corr()


# In[29]:


# 通过属性的相关关系来填补缺失值
target_data = data['1_DAY_RETURN'].copy(deep=True)
source_data = data['LSTM_POLARITY'].copy(deep=True)
flag1 = target_data.isnull().values
flag2 = source_data.isnull().values
i=0
for _,value in target_data.iteritems():
    if(flag1[i]==True) and (flag2[i]==False):
        target_data[i] = 3 - source_data[i]
    i=i+1


# In[30]:


# 1_DAY_RETURN 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("1_DAY_RETURN hist")
data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new 1_DAY_RETURN hist")
target_data.hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("1_DAY_RETURN box")
data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new 1_DAY_RETURN box")
target_data.plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['1_DAY_RETURN'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(target_data,dist="norm",plot=plt)
plt.show()


# In[31]:


target_data.describe()#查看数据描述


# ### 3.4 通过对象的相似性填补缺失值

# In[39]:


numeric_attr = ['1_DAY_RETURN','2_DAY_RETURN']

# 相似性选择
def find_dis_value(dataset, pos, numeric_attr):
    def dis_objs(tar_obj_index, sou_obj_index):
        tar_obj = dataset.iloc[tar_obj_index]
        sou_obj = dataset.iloc[sou_obj_index]
        dis_value = 0
        for column in tar_obj.index:
            if column == 'Priority':
                if (not math.isnan(tar_obj[column])) and (not math.isnan(sou_obj[column])):
                    dis_value += sou_obj[column] - tar_obj[column]
                else:
                    dis_value += 9998
        return dis_value
    mindis = 9999
    result_pos = -1
    leftindex = 0;
    rightindex = dataset.shape[0]-1
    # 二分查找返回最近距离的一个 result_pos
    while leftindex<=rightindex:
        midindex = int((leftindex+rightindex)/2)
        tmpdis = dis_objs(pos,midindex)
        if(tmpdis>0):
            rightindex = midindex-1
        elif(tmpdis == 0):
            result_pos = midindex
            break;
        else:
            leftindex = midindex+1
        if(tmpdis<mindis):
            result_pos = midindex
    return result_pos
# 通过数据对象之间的相似性来填补缺失值
numical_datasets = pd.DataFrame(data[numeric_attr].copy(deep=True))

# 对 numical_datasets 排序
numical_datasets.sort_values("2_DAY_RETURN",inplace=True)
data_day_return = numical_datasets['1_DAY_RETURN'].copy(deep=True)
print('空数据数量为:',data_day_return.isnull().sum())
length = numical_datasets.shape[0]
count=1;
for i in range(length):
    if math.isnan(numical_datasets['1_DAY_RETURN'].iloc[i]):
        result_pos = find_dis_value(numical_datasets, i, numeric_attr)
        data_day_return.iloc[i] = data_day_return.iloc[result_pos]
        count+=1


# In[40]:


# 填充后的空数据数量
print(data_day_return.isnull().sum())


# In[41]:


# 1_DAY_RETURN 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("1_DAY_RETURN hist")
data['1_DAY_RETURN'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new 1_DAY_RETURN hist")
data_day_return.hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("1_DAY_RETURN box")
data['1_DAY_RETURN'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new 1_DAY_RETURN box")
data_day_return.plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['1_DAY_RETURN'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(data_day_return,dist="norm",plot=plt)
plt.show()

