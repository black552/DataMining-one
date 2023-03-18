#!/usr/bin/env python
# coding: utf-8

# # 数据挖掘

# ## 姓名：刘思雯 学号：3120220948

# ## 1.数据集GitHub Dataset

# In[2]:


# 导入必要的包
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from math import isnan
import math


# ### 1.1查看数据集

# In[2]:


#查看数据集下的数据文件
import os
for dirname, _, filenames in os.walk('.\data\GitHub Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#数据集文件包含github_dataset和repository_dataset两个数据集
#这里以github_dataset数据集为例进行研究


# ## 1.2读取数据集

# In[4]:


#读取数据集
path = './data/GitHub Dataset/repository_data.csv'
data = pd.read_csv(path,low_memory=False)
data.head()# 默认展示前五行数据


# In[15]:


data.dtypes # 每列数据的数据类型


# In[16]:


data.shape # 数据集的大小


# ## 2.数据分析要求

# ### 2.1数据摘要

# 数据集各属性含义：
# * name - the name of the repository
# * stars_count - stars count of the repository
# * forks_count - forks count of the repository
# * watchers - watchers in the repository
# * pull_requests - pull requests made in the repository
# * primary_language - the primary language of the repository
# * languages_used - list of all the languages used in the repository
# * commit_count - commits made in the repository
# * created_at - time and date when the repository was created
# * license - license assigned to the repository.

# 2.1.1 标称属性

# In[20]:


# 由上面对数据集分析可知，该数据集的标称属性有"name"、"primary_language"、
#"languages_used"、"created_at"、"license"两个标称属性
# 下面给出每个属性取值的频数
#（1）name
pd.value_counts(data['name'])


# In[21]:


#（2）primary_language
pd.value_counts(data['primary_language'])


# In[22]:


#(3)languages_used
pd.value_counts(data['languages_used'])


# In[23]:


#(4)created_at
pd.value_counts(data['created_at'])


# In[25]:


#（5）licence
pd.value_counts(data['licence'])


# 2.1.2 数值属性

# In[26]:


# 这里的数值属性包括 stars_count、forks_count、issues_count、pull_requests和 contributors
# 对数值属性的 5 数进行概括
digital_data = ['stars_count','forks_count','watchers','pull_requests','commit_count']
data[digital_data].describe()


# In[18]:


#给出各数值属性缺失值个数
print("'stars_count'：",data['stars_count'].isnull().sum())
print("'forks_count'：",data['forks_count'].isnull().sum())
print("'watchers'：",data['watchers'].isnull().sum())
print("'pull_requests'：",data['pull_requests'].isnull().sum())
print("'commit_count'：",data['commit_count'].isnull().sum())


# 由此可见，只有commit_count存在1921个缺失值。

# ### 2.2数据可视化

# （1）绘制stars_count的直方图、盒图、qq图

# In[34]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("stars_count hist")
data['stars_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("stars_count box")
data['stars_count'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(2,2,3)
stats.probplot(data['stars_count'],dist="norm",plot=plt)
# 绘制 q-q 图
plt.subplot(2,2,4)
stats.probplot(data['stars_count'], dist="norm", plot=plt)
plt.show()


# (2)绘制forks_count的直方图、盒图、qq图

# In[43]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("forks_count hist")
data['forks_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("forks_count box")
data['forks_count'].plot(kind='box',notch=True,grid=True)
# 绘制 q-q 图
plt.subplot(2,2,3)
stats.probplot(data['forks_count'], dist="norm", plot=plt)
plt.show()


# (3)绘制watchers的直方图、盒图、qq图

# In[44]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("watchers hist")
data['watchers'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("watchers box")
data['watchers'].plot(kind='box',notch=True,grid=True)
# 绘制 q-q 图
plt.subplot(2,2,3)
stats.probplot(data['watchers'], dist="norm", plot=plt)
plt.show()


# (4)绘制pull_requests的直方图、盒图、qq图

# In[45]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("pull_requests hist")
data['pull_requests'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("pull_requests box")
data['pull_requests'].plot(kind='box',notch=True,grid=True)
# 绘制 q-q 图
plt.subplot(2,2,3)
stats.probplot(data['pull_requests'], dist="norm", plot=plt)
plt.show()


# (5)绘制commit_count的直方图、盒图、qq图

# In[46]:


# coding=utf-8
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(2,2,1)
plt.title("commit_count hist")
data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(2,2,2)
plt.title("commit_count box")
data['commit_count'].plot(kind='box',notch=True,grid=True)
# # 去除缺失值再绘制 q-q 图
plt.subplot(2,2,3)
data_drop=pd.DataFrame(data['commit_count'].copy(deep=True))
data_drop = data_drop.dropna()
stats.probplot(data_drop['commit_count'], dist="norm", plot=plt)
plt.show()


# ## 3.数据缺失值处理

# ### 3.1剔除缺失值

# In[6]:


#统计缺失值
def missing_data(datatodel):
    missing_num = datatodel.isnull().sum()
    missing_percent = missing_num/datatodel.shape[0]*100
    concat_data = pd.concat([missing_num,missing_percent],axis=1,keys=['missing_num','missing_percent'])
    concat_data['Types'] = datatodel.dtypes
    return concat_data
missing_data(data)


# 由上表可以看出，数值属性commit_count存在缺失值
# 标称属性name、prinary_language、languages_used、licence存在缺失值
# 可能原因是未记录或者无法获取

# In[7]:


del_null_data = data.copy(deep=True)
del_null_data = del_null_data.dropna()
# 处理缺失数据后的数据展示
missing_data(del_null_data)


# In[8]:


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# commit_count 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("commit_count hist")
data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new commit_count hist")
del_null_data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("commit_count box")
data['commit_count'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new commit_count box")
del_null_data['commit_count'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['commit_count'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(del_null_data['commit_count'],dist="norm",plot=plt)
plt.show()


# In[57]:


del_null_data['commit_count'].describe() # 缺失部分剔除后数据的 5 数概况


# ### 3.2 用最高频率填补缺失值

# In[9]:


# 用最高频率来填补缺失值--此处使用深拷贝，否则会改变原值
fill_data_with_most_frequency = data.copy(deep=True)
# 对'commit_count' 进行最高频率值填补缺失值
word_counts = Counter(fill_data_with_most_frequency['commit_count'])
top = word_counts.most_common(1)[0][0]
fill_data_with_most_frequency['commit_count'] = fill_data_with_most_frequency['commit_count'].fillna(top)
# 查看填充后是否还有数据缺失
missing_data(fill_data_with_most_frequency)


# In[10]:


# commit_count 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("commit_count hist")
data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new commit_count hist")
fill_data_with_most_frequency['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("commit_count box")
data['commit_count'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new commit_count box")
fill_data_with_most_frequency['commit_count'].plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['commit_count'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(fill_data_with_most_frequency['commit_count'],dist="norm",plot=plt)
plt.show()


# In[60]:


# 对填充后的新数据进行描述
fill_data_with_most_frequency['commit_count'].describe()


# ### 3.3通过属性的相关关系来填补缺失值

# In[61]:


# 查看相关的属性关系
data.corr()


# In[12]:


# 通过属性的相关关系来填补缺失值
target_data = data['commit_count'].copy(deep=True)
source_data = data['pull_requests'].copy(deep=True)
flag1 = target_data.isnull().values
flag2 = source_data.values
i=0
for _,value in target_data.iteritems():
    if(flag1[i]==True) and (flag2[i]==False):
        target_data[i] = 3 - source_data[i]
        i=i+1


# In[13]:


# commit_count 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("commit_count hist")
data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new commit_count hist")
target_data.hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("commit_count box")
data['commit_count'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new commit_count box")
target_data.plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['commit_count'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(target_data,dist="norm",plot=plt)
plt.show()


# In[65]:


# 对填充后的新数据进行描述
target_data.describe()


# ### 3.4 通过对象的相似性填补缺失值

# In[14]:


numeric_attr = ['commit_count','pull_requests']

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
numical_datasets.sort_values("pull_requests",inplace=True)
data_commit_count = numical_datasets['commit_count'].copy(deep=True)
print('空数据数量为:',data_commit_count.isnull().sum())
length = numical_datasets.shape[0]
count=1;
for i in range(length):
    if math.isnan(numical_datasets['commit_count'].iloc[i]):
        result_pos = find_dis_value(numical_datasets, i, numeric_attr)
        data_commit_count.iloc[i] = data_commit_count.iloc[result_pos]
        count+=1


# In[69]:


# 填充后的空数据数量
print(data_commit_count.isnull().sum())


# In[15]:


# commit_count 可视化对比新旧数据
plt.figure(figsize = (10,10))
# 直方图
plt.subplot(3,2,1)
plt.title("commit_count hist")
data['commit_count'].hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 直方图
plt.subplot(3,2,2)
plt.title("new commit_count hist")
data_commit_count.hist(alpha=0.5,bins=15) #alpha 透明度，bins 竖条数
# 盒图
plt.subplot(3,2,3)
plt.title("commit_count box")
data['commit_count'].plot(kind='box',notch=True,grid=True)
# 盒图
plt.subplot(3,2,4)
plt.title("new commit_count box")
data_commit_count.plot(kind='box',notch=True,grid=True)
#q-q 图
plt.subplot(3,2,5)
stats.probplot(data['commit_count'],dist="norm",plot=plt)
plt.subplot(3,2,6)
stats.probplot(data_commit_count,dist="norm",plot=plt)
plt.show()


# In[71]:


# 对填充后的新数据进行描述
data_commit_count.describe()


# In[ ]:




