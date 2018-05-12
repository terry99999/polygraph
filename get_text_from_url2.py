import newspaper

def url_to_string(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    return article.text