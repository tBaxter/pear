# start by pulling the python image
FROM python:3.8-alpine as base
# switch working directory
WORKDIR /app
EXPOSE 8000
#RUN apt-get update

# copy the requirements file into the image and install requirements.
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# initialize NLTK
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

# copy every content from the local file to the image
COPY . .

CMD ["gunicorn", "--reload", "--bind=0.0.0.0:8000", "app:app"]
