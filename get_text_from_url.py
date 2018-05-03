from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib
import urllib.request
import re
import ssl

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def url_to_string(url):
    context = ssl._create_unverified_context()
    html = urllib.request.urlopen(url, context=context).read()
    return (' '.join(text_from_html(html).split()))

def string_to_file(file_name, s):
    textfile = open(file_name, 'w')
    textfile.write(s)
    textfile.close()
