# pear
Simple Flask app to compare the text of two documents using Python's Natural Language Tool Kit

### To run locally:
Pipenv is recommended. If you don't have it already installed, then
`pip install pipenv`

Install dependencies:
`pipenv install`

Then:
```
pipenv shell
flask run
```
Then -- and this is an important step -- before you use the form, be sure to download and update the NLTK dependencies. You can do this by going to /update-nltk in the running app.