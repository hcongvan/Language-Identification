import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


data = pd.read_csv('formatted_data.csv', sep=';', error_bad_lines=False)
data.head()

data.describe()
test = data.values

vectorizer = TfidfVectorizer(analyzer='char')
X = vectorizer.fit_transform([test[0][1]])
print(vectorizer.get_feature_names())

a = CountVectorizer(analyzer='char')
y = a.fit_transform([test[0][1]])
g = a.vocabulary_

count = np.sum(y,1)