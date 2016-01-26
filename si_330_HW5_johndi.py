# -*- coding: utf-8 -*-
import nltk
from nltk.stem import *
from nltk.corpus import stopwords
import tweepy
import json
import time
import vincent
import re
import pandas
import csv
consumer_key = 'ofd1WyrA0azSMsYWGPnoTjuCp'
consumer_secret = 'Elz1EKDm4NT5rkNqnw4RikbDq2cgtS3sZ6sQET0tPolV150pBn'
access_token = '1525418455-3ub7PXQgU3itK7V6dXqMhrKdZ1xSNXT7r5ZcSRv'
access_token_secret = 'aa2wbzaYUprDxfY77hUsttF2tlEFPyQJGtohcZEVgIgzc'

start_time = time.time()
stemmer = PorterStemmer()
stopwords = stopwords.words('english')

class listener(tweepy.StreamListener):
    def __init__(self, start_time, time_limit):
        self.time = start_time
        self.limit = time_limit
        self.tweet_data = []

    def on_data(self,data):
        if(time.time() - self.time) < self.limit:
            try:
                decoded = json.loads(data)
                text = decoded['text']
                date = decoded['created_at']
                with open('si_330_johndi_rawtweets.json','a+') as outfile:
                    # json.dump({'tweet':text,'date':date},outfile, sort_keys=True)
                    outfile.write(json.dumps({'tweet':text,'date':date}).encode('utf-8'))
                    # outfile = outfile.encode('utf-8')
                    outfile.write('\n')
                text = preprocess(text)
                with open('si_330_johndi_cleanedtweets.json','a+') as outfile2:
                    outfile2.write(json.dumps({'tweet':text,'date':date}).encode('utf-8'))
                    outfile2.write('\n')
                    # outfile2 = outfile2.encode('utf-8')
                self.tweet_data.append(pair)
                return True
            except:
                #print "failed on_data"
                return True
        return False

    def on_error(self,status):
        print status


def preprocess(text):
    text = text.lower()
    text = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', 'URL', text)
    text = re.sub(r'@[A-Za-z|0-9]+', 'AT_USER', text)
    text = re.sub(r'#([A-Za-z|0-9])+', '\1', text)
    text = re.sub(r'\.', ' . ', text)
    text = re.sub(r'\!', ' ! ', text)
    text = re.sub(r'\?', ' ? ', text)
    text = re.sub(r'\,', ' , ', text)
    text = re.sub(r'\:', ' : ', text)
    text = re.sub(r'\[', ' [ ', text)
    text = re.sub(r'\(', ' ( ', text)
    text = re.sub(r'\]', ' ] ', text)
    text = re.sub(r'\)', ' ) ', text)
    text = re.sub(r'\"', ' ', text)
    text = re.sub(r'\,+', ' ', text)
    text = re.sub(r'\!+', ' ', text)
    text = re.sub(r'\?+', ' ', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\\', '', text)
    return text


def extract_features(input_list):
    counts_dict = {}
    for item in input_list:
        item = stemmer.stem(item)
        if item not in counts_dict:
            counts_dict[item] = 1
        else:
            counts_dict[item] += 1
    return counts_dict


def train_set(training_data):
    csvfile = csv.reader(open(training_data,'rU'), delimiter=',')
    csvfile.next()
    list_of_tuples = []
    for row in csvfile:
        text = row[0]
        row = [preprocess(item) for item in row]
        sentiment = row[2]
        sentiment = sentiment.lower().rstrip()
        text = text.split()
        text_bigrams = get_ngrams(text,2)
        for item in text_bigrams:
            text.append(item)
        features = extract_features(text)
        with_label = (features,sentiment)
        # print with_label
        list_of_tuples.append(with_label)
    classifier = nltk.NaiveBayesClassifier.train(list_of_tuples)
    # print classifier.most_informative_features
    return classifier

def get_ngrams(input, n):
    empty = []
    tuple = zip(*[input[i:] for i in range(n)])
    for item in tuple:
        tuple_string = u'+'.join(item[0:1])
        # tuple_string = str(item[0]) + " + " + str(item[1])
        empty.append(tuple_string)
    return empty

def sentiment_analysis(classifier):
    with open('si_330_johndi_rawtweets.json','rU') as infile:
        for line in infile:
            line = json.loads(line)
            text = line['tweet']
            date = line['date']
            text_array_uni = text.split()
            text_array_bigrams = get_ngrams(text_array_uni,2)
            # print text_array_uni , type(text_array_uni)
            # print text_array_bigrams , type(text_array_bigrams)
            # text_array_uni = text_array_uni + text_array_bigrams
            text_array_uni = [preprocess(line) for line in text_array_uni if line not in stopwords]
            features = extract_features(text_array_uni)
            label = classifier.classify(features)
            with_label = (features,label)
            with open('si_330_johndi_labelledtweets.json','a+') as outfile:
                outfile.write(json.dumps({'tweet':text,'date':date,'sentiment':label}).encode('utf-8'))
                # outfile = outfile.encode('utf-8')
                outfile.write('\n')


def time_series_tweets():
    with open('si_330_johndi_labelledtweets.json') as graph_file:
        d = dict()
        time_array = []
        for line in graph_file:
            data = json.loads(line)
            time_array.append(data['date'])

    ones = [1]*len(time_array)

    idx = pandas.DatetimeIndex(time_array)

    t_array = pandas.Series(ones, index=idx)

    t_array = t_array.resample('5s', how='sum').fillna(0)
    time_chart = vincent.Line(t_array)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json('term_freq.json')

def time_series_tweets2():
    with open('si_330_johndi_labelledtweets.json') as graph_file:
        d = dict()
        time_array = []
        for line in graph_file:
            data = json.loads(line)
            if data['sentiment'] == "positive":
                time_array.append(data['date'])

    ones = [1]*len(time_array)

    idx = pandas.DatetimeIndex(time_array)

    t_array = pandas.Series(ones, index=idx)

    t_array = t_array.resample('5s', how='sum').fillna(0)
    time_chart = vincent.Line(t_array)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json('term_freq2.json')

if __name__ == "__main__":
    l = listener(start_time, time_limit=20)
    #Specify credentials
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    print "Receiving and processing twitter stream, please wait...."
    # l = dir(__builtins__)
    # print l
    # In this example we follow #monday hashtag
    stream = tweepy.Stream(auth, l)
    # Listen to Twitter streaming data for the given keyword. Narrow it down to English.
    stream.filter(track='#sarcasm', languages=['en'])
    print "Training classifier..."
    classifier = train_set('training_set.csv')
    print "Conducting sentiment analysis..."
    sentiment_analysis(classifier)
    time_series_tweets()
    time_series_tweets2()
