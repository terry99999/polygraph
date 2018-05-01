import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as skt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import eli5
from eli5.lime import TextExplainer
from sklearn.datasets import fetch_20newsgroups


data = pd.read_csv("fake_or_real_news.csv")

data = data.iloc[:,2:4]
data["label"].replace({"FAKE":"0", "REAL":"1"}, inplace=True)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def print_report(pipe, test_values, test_targets):
    y_test = test_targets
    y_pred = pipe.predict(test_values)
    report = metrics.classification_report(y_test, y_pred, target_names=("UNRELIABLE", "RELIABLE"))
    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))



#classifier = MultinomialNB()
#count_vectorizer.fit(data["text"])
#feature_names = count_vectorizer.get_feature_names()
#tokens_with_weights = sorted(zip(classifier.coef_[0], feature_names))[:20]
#counts = count_vectorizer.fit_transform(data["text"].values)
#feature_names = count_vectorizer.get_feature_names()


kf = KFold(3)
for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_targets = data.iloc[train_index,1].values
    train_values = data.iloc[train_index,0].values
    test_targets = data.iloc[test_index,1].values
    test_values = data.iloc[test_index,0].values
    vec = TfidfVectorizer(stop_words=skt.ENGLISH_STOP_WORDS)
    clf = LogisticRegressionCV()
    pipe = make_pipeline(vec, clf)
    pipe.fit(train_values, train_targets)
    print_report(pipe, test_values, test_targets)
    print(eli5.format_as_text(eli5.explain_weights(clf, vec=vec, target_names=("UNRELIABLE", "RELIABLE"))))

#counts = count_vectorizer.fit_transform(data["text"].values)
#classifier.fit(counts,data.iloc[:,1].values)
#pizza_data = pd.read_csv("pizzagate.csv")
#test_counts = count_vectorizer.transform(pizza_data["text"].values)
#print(pizza_data)
#print(classifier.classes_)
#print(classifier.coef_)
#print(classifier.predict_proba(test_counts))

