# pear
Pear is a simple Flask app that compares the text of two documents -- like a resume and a job description -- using Python's Natural Language Tool Kit. It is online at https://pear.fly.dev

Pear requires no sign-up, does not harvest or store your data, and is completely free and open. It began sinmply because I was job hunting and I wanted to experiment with Flask, with HTMX and with Natural Language Processing and Python's Natural Language ToolKit. This was the result. 

It is not well-tested and my understanding of Natural Language Processing is not particularly sophisticated. No warranty is expressed or implied. 

Pull requests, however, are welcomed.

### To run online:
Go to https://pear.fly.dev

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
