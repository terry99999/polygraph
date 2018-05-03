import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_name):
    with open(file_name, 'r') as input_file:
        s = ''
        for line in input_file:
            s += line
    return s

rootdir = os.fsencode('/Users/bettychou1993/Desktop/polygraph/story_db')

def get_dir_cat(rootdir):
    cat_and_path = {}
    for subdir, dirs, files in os.walk(rootdir):
        if subdir != rootdir:
            cat = os.path.basename(subdir).decode("utf-8")
            cat_and_path[cat] = []
            for f in files:
                filepath = os.path.join(subdir, f)
                cat_and_path[cat].append(filepath.decode("utf-8"))
    return cat_and_path

#article = load_data('article.txt')
def match_articles(unknown_article):
    vect = TfidfVectorizer(min_df=1)
    matches = []
    for c, path in get_dir_cat(rootdir).items():
        s = ''
        for p in path:
            s += load_data(p)
        tfidf = vect.fit_transform([unknown_article, s])
        matches.append(((tfidf * tfidf.T)[0, 1], c))
    return matches, max(matches)[1]