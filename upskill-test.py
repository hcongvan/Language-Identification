import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy

data = pd.read_csv('formatted_data.csv', sep=';', error_bad_lines=False)
data.head()

data.describe()
data = data.values
n = data.shape[0]
corpus = ''
# for i in data:
#     corpus = corpus.join(i[1])
vector = CountVectorizer(analyzer='char',encoding='latin-1',ngram_range=(2,2))
language_train = []
language_test = []
string = []
X_test = np.array([None]*n)
y_test = np.array([None]*n)
vocabulary = np.array([None]*n)
for i,v in enumerate(data):
    string.append(v[1])
    _tmp_X = np.array(v[1].split('.'))
    _tmp_y = np.array([v[0]]*_tmp_X.shape[0])
    X_train, X_test[i], _, y_test[i] = train_test_split(_tmp_X, _tmp_y, test_size=0.2, random_state=42)
    corpus = ''.join(X_train)
    v[1] = corpus

    # vector.fit([corpus.replace(' ','')])
    # print(len(vector.vocabulary_))
    # vocabulary[i] = vector.vocabulary_
    # y = vector.transform([corpus.replace(' ','')])
    # language_train[i] = np.array([corpus,v[0]]).reshape([1,2])

y_test = y_test[:][0]
vector.fit(string)
m = len(vector.vocabulary_)
for i,v in enumerate(data):
    transform = vector.transform([v[1]])
    _q = np.zeros(m)
    for k,data in zip(transform.indices,transform.data):
            _q[k] = data
    language_train.append(np.array([transform,_q,v[0]]))

for i,v in enumerate(X_test):
    corpus = ''.join(v)
    transform = vector.transform([corpus.replace(' ','')])
    _p = np.zeros(m)
    for k,data in zip(transform.indices,transform.data):
            _p[k] = data
    language_test.append(np.array([transform,_p,y_test[i]]))
prediction = np.zeros(n)
for i in range(n):
    t = np.zeros(n)
    for j in range(n):
        t[j] = sklearn.metrics.mutual_info_score(language_test[i][1],language_train[j][1])
    prediction[i] = np.argmin(t)
    
# for i,v in enumerate(zip(X_test,y_test)):
# corpus1 = ''.join(X_test[0])
# test1 = CountVectorizer(analyzer='char',encoding='latin-1')
# q1 = test1.fit_transform([corpus1.replace(' ','')])

# corpus2 = ''.join(X_test[1])
# test2 = CountVectorizer(analyzer='char',encoding='latin-1')
# q2 = test2.fit_transform([corpus2.replace(' ','')])

# corpus3 = ''.join(X_test[2])
# test3 = CountVectorizer(analyzer='char',encoding='latin-1')
# q3 = test3.fit_transform([corpus3.replace(' ','')])

# rus = language_train[0][0][1]
# rus_p = language_train[0][0][0]
# text = corpus1.replace(' ','')
# test = rus.transform([text])

# p = [0]*138
# q = [0]*138
# for i,data in zip(rus_p.indices,rus_p.data):
#     p[i] = data
# for i,data in zip(q3.indices,q3.data):
#     q[i] = data
# t = scipy.stats.entropy(q,p)
print("TEST")
# scipy.stats.entropy(q3.data,rus_p.data)