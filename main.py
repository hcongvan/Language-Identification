# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
import scipy
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# <h2>Function replace some symbol</h2>
# Follow table below we replace some character to 1 symbol, because all of them resentation the popular word, such as english token [a-zA-Z] 
# <img src="img/table1.png"/>

# %%
def removeNoise(data):
    data['text'] = data['text'].str.replace('[=<>"[]{}/:-;.,\'\(\)%]', ' ')
    data['text'] = data['text'].str.replace('[0-9]', ' ')
    return(data)

# %% [markdown]
# <h2>Formulate KL-divergence</h2>
# 
# - Make sure 2 params are array and same length
# - Normalize them 
# 
# <img src="img/kl.png" />

# %%
def KLdivergence(p,q):
    if len(p) != len(q):
        return False
    p = p/np.sum(p)
    q = q/np.sum(q)
    return np.sum(p*np.log(p/q))

# %% [markdown]
# <h2>Get data train</h2> 

# %%
data = pd.read_csv('formatted_data.csv', sep=';', error_bad_lines=False)
data = removeNoise(data)
data.head()

# %% [markdown]
# <h2>Get data test from external test set</h2>
# With data test:
# 
# - Using other source but keep format and language support
# - Select random line for test

# %%
test = pd.read_csv('europarl.csv',sep=';', error_bad_lines=False)
test = removeNoise(test)
test= test.reindex(np.random.permutation(test.index))
test.head()

# %% [markdown]
# <h2>Prepair data</h2>
# 
# Transform data and test set to numpy array with : 
# - n: length of data train
# - m: length of data test
# 
# List of all the languages whose detection is supported:
# 
# - 'bg': Bulgarian
# - 'cs': Czech
# - 'da': Danish
# - 'de': German
# - 'el': Greek, Modern
# - 'en': English
# - 'es': Spanish
# - 'et': Estonian
# - 'fi': Finnish
# - 'fr': French
# - 'hu': Hungarian
# - 'it': Italian
# - 'lt': Lithuanian
# - 'lv': Latvian
# - 'nl': Dutch
# - 'pl': Polish
# - 'pt': Portuguese
# - 'ro': Romanian
# - 'sk': Slovak
# - 'sl': Slovenian
# - 'sv': Swedish

# %%
data = data.values
test = test.values

n = data.shape[0]
m = test.shape[0]

labels = data[:,0]
labels_test = test[:,0]

# %% [markdown]
# Prepair data train:
# - using CountVectorizer to get all token from raw text
# - Seperate each token language 

# %%
language_train = []
for i,v in enumerate(data):
    vector = CountVectorizer(analyzer='char',encoding='latin-1',ngram_range=(2,2))
    y = vector.fit_transform([v[1].replace(' ','')])
    language_train.append(np.array([vector,y]))

# %% [markdown]
# Predict task:
# 
# - Get each text from test data
# - Get token don't exit from _p with _q assign to _set(p is set token of test, and q is set token of train)
# <img src="img/t_cup.png"/>
# - If _set empty => _p inside _q then calculate KL-divergence D(_p||_q) will > 0
# <img src="img/t_cap.png"/>
# - Else _set have some character => _p overlap _q the D(_p||_q) will Infinity
# - Store D(_p||_q) into t array
# - After run all language support, predict = argmin(t) 

# %%
predict_label = []
true_position = 0
for i,(_,data_test) in enumerate(test):
    vector_test = TfidfVectorizer(analyzer='char',encoding='latin-1',ngram_range=(2,2))
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


# %%
print(classification_report(labels_test, predict_label))            
print (accuracy_score(labels_test, predict_label))

