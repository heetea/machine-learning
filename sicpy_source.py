#!/usr/bin/env python
# coding: utf-8

# In[5]:


#소스코드
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
# csv 파일로 불러오기
csv = pd.read_csv("C:/Users/sd33/iris.csv")
# 데이터프레임 만들기
df = pd.DataFrame(csv)


# In[6]:


### 결측데이터, NaN데이터 처리
clean_df = df.fillna(df.mean()) # NaN을 처리해줌, Missing, 0, df.mean()
clean_df[clean_df=='']='Unknown' # str type이지만, 없는 데이터를 처리해줌
# 데이터와 레이블로 분류하기
X = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y = df[["Name"]]
#트레이닝 테스트로 분류
X_train, X_test, y_train, y_test = train_test_split(X, y)
#기술통계
df.describe(include='all')


# In[ ]:


print(df.nunique())


# In[ ]:


df['SepalLength'].value_counts().plot.bar(title="SepalLength")


# In[ ]:


df['SepalWidth'].value_counts().plot.bar(title="SepalWidth")


# In[ ]:


df['PetalLength'].value_counts().plot.bar(title="PetalLength")


# In[ ]:


df['PetalWidth'].value_counts().plot.bar(title="PetalWidth")


# In[ ]:


df_hist = df[df['Name'] == 0]
df = df[df['Name'] == 1]
#시각화 plot, scatter, hist
col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

fig, ax = plt.subplots(len(col_names), figsize=(16,12))

for i, col_val in enumerate(col_names):

    sns.distplot(df_hist[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)

plt.show()


# In[ ]:


sns.pairplot(df_hist)


# In[7]:


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# In[10]:


for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    results = clf.predict(X_test)#학습된 모델로 결과값 도출

    score = metrics.accuracy_score(results, y_test)#결과값 정답률 확인하기
    print(name, "정답률:", score) 


# In[ ]:




