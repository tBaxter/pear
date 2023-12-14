# pear
Pear is a simple Flask app that compares the text of two documents -- like a resume and a job description -- using Python's Natural Language Tool Kit. 

It began because I was job hunting and I wanted to experiment with Flask, with HTMX and with the NLTK. This was the result. 

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
Then -- and this is an important step -- before you use the form, be sure to download and update the NLTK dependencies. You can do this by going to `/update-nltk` in your browser address bar when app is running.
