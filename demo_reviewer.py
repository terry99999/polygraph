from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import get_text_from_url2
import os
import users
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
from IPython.display import display
from IPython.display import HTML
import get_text_from_url
import newspaper as news

from sklearn import metrics


def print_report(pipe, test_values, test_targets):
    y_test = test_targets
    y_pred = pipe.predict(test_values)
    report = metrics.classification_report(y_test, y_pred, target_names=("UNRELIABLE", "RELIABLE"))
    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))


def evaluate_url(url):
    article = news.Article(url)
    article.download()
    article.parse()
    print(article.summary)
    display(eli5.show_prediction(clf, article.text, vec=vec, target_names=("UNRELIABLE", "RELIABLE")))


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
rootdir = os.fsencode('/Users/Daniel/PycharmProjects/polygraph')


class ReusableForm(Form):
    url = TextField('Please enter an valid url:', validators=[validators.required()])


data = pd.read_csv("fake_or_real_news.csv")

data = data.iloc[:, 2:4]
data["label"].replace({"FAKE": "0", "REAL": "1"}, inplace=True)
vec = CountVectorizer(stop_words=skt.ENGLISH_STOP_WORDS, ngram_range=(2,2))
clf = LogisticRegressionCV()
pipe = make_pipeline(vec, clf)
pipe.fit(data.iloc[:, 0].values, data.iloc[:, 1].values)

@app.route("/", methods=['GET', 'POST'])
def hello():

    form = ReusableForm(request.form)

    print(form.errors)
    if request.method == 'POST':
        url = request.form['url']
        print(url)

        if form.validate():
            if url == "model":
                flash(eli5.show_weights(clf, vec=vec, target_names=("UNRELIABLE","RELIABLE")))
            else:
                # Save the comment here.
                article = news.Article(url)
                article.download()
                article.parse()
                flash('show the article analysis:')
                flash(eli5.show_prediction(clf, article.text, vec=vec, target_names=("UNRELIABLE", "RELIABLE")))
        else:
            flash('All the form fields are required. ')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()