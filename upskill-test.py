import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
import scipy

def removeNoise(data):
    data['text'] = data['text'].str.replace('[=<>"{}/:-;.,\'\(\)%]', ' ')
    data['text'] = data['text'].str.replace('[0-9]', ' ')
    return(data)

data = pd.read_csv('formatted_data.csv', sep=';', error_bad_lines=False)
data = removeNoise(data)
test = pd.read_csv('europarl.csv',sep=';', error_bad_lines=False)
test = removeNoise(test)
test= test.reindex(np.random.permutation(test.index))

data = data.values
test = test.values

n = data.shape[0]
m = test.shape[0]

labels = data[:,0]
labels_test = test[:,0]

language_train = []
for i,v in enumerate(data):
    vector = CountVectorizer(analyzer='char',encoding='latin-1',ngram_range=(2,2))
    y = vector.fit_transform([v[1].replace(' ','')])
    print(len(vector.vocabulary_))
    language_train.append(np.array([vector,y]))




prediction = np.zeros([m],dtype=int)
predict_label = []
true_position = 0
for i,(_,data_test) in enumerate(test):
    vector_test = CountVectorizer(analyzer='char',encoding='latin-1',ngram_range=(2,2))
    transform = vector_test.fit_transform([data_test.replace(' ','')])
    t = np.array([float('Inf')]*n)
    for j,(vector_train,transform_y) in enumerate(language_train):
        _tmp = set(vector_train.vocabulary_)^set(vector_test.vocabulary_)
        _set = list((set(_tmp)|set(vector_test.vocabulary_))^ set(vector_train.vocabulary_))
        if not bool(_set):
            k = len(vector_train.vocabulary_)
            _q = np.zeros(k)
            _q[transform_y.indices] = transform_y.data
            _p = np.zeros(k)
            _p[transform.indices] = transform.data
            t[j] = scipy.stats.entropy(_p,_q)
    predict_label.append(labels[np.argmin(t)])
    

# accuracy_score = true_position/m   
print(classification_report(labels_test, predict_label))            
print (accuracy_score(labels_test, predict_label))
