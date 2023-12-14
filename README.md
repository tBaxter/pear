# pear
Pear is a simple Flask app to compare the text of two documents using Python's Natural Language Tool Kit. For example, it can effectively compare your resume with a given job description and give you a sense of how they match. It began because I was job hunting and I wanted to experiment with Flask, with HTMX and with the NLTK, and this was the result. 

It is not well-tested and my understanding of Natural Language Processing is not sophisticated. Use it with that in mind and no warranty is expressed or implied. Pull requests, however, are welcomed.


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
