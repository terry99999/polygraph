from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_name):
    with open(file_name, 'r') as input_file:
        s = ''
        for i, line in enumerate(input_file):
            if line != '\n':
                s += line.rstrip()
    return s

print 'read in target article and other stories'

article = load_data('article.txt')
story_1 = load_data('story_1.txt')
story_2 = load_data('story_2.txt')
story_3 = load_data('story_3.txt')

print 'transfer text documents into tf-idf vectors'
vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform([article, story_1, story_2, story_3])

print 'compute cosine similarity between article and stories'
print (tfidf * tfidf.T).A

print 'higher numbers means higher similarity'