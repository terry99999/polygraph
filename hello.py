from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import get_text_from_url
import os
import users

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
rootdir = os.fsencode('/Users/bettychou1993/Desktop/polygraph/story_db')

class ReusableForm(Form):
    url = TextField('Please enter an valid url:', validators=[validators.required()])
    
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print (form.errors)
    if request.method == 'POST':
        url=request.form['url']
        print (url)
 
        if form.validate():
            # Save the comment here.
            unknown = get_text_from_url.url_to_string(url)
            sim, m = users.match_articles(unknown)
            flash('show the similarities among unknown articles and stories:')
            flash(sim)
            flash('show the matches:')
            flash(m)
            flash(type(m))
            flash(users.get_dir_cat(rootdir)[m])
        else:
            flash('All the form fields are required. ')
 
    return render_template('hello.html', form=form)
 
if __name__ == "__main__":
    app.run()