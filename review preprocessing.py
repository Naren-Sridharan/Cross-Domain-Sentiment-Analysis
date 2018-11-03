import zipfile
import csv
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil import parser
from pprint import pprint


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    words = tokenizer.tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    filtered_words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    return filtered_words


with open('1429_1.csv', 'r', encoding="latin1") as amazon_reviews:
    avg = 0
    count = 0
    for row in csv.reader(amazon_reviews):
        if row[0] == 'id':
            continue
        avg += float(3.0 if row[14] == '' else row[14])
        count += 1
    avg /= count

print('Average rating for amazon reviews: ' + str(avg))

with open('1429_1.csv', 'r', encoding="latin1") as amazon_reviews:
    with open('amazon_data.pickle', 'wb') as pickle_file:
        reviews = list(dict())
        i = 0
        j = 0
        for row in csv.reader(amazon_reviews):
            if row[0] == 'id':
                continue
            tokens = preprocess(row[16])
            sentiment = float(3 if row[14] == '' else row[14]) >= (avg - 0.5)

            if (i < 4000 and sentiment) or (j < 4000 and not sentiment):
                reviews.append({'tokens': tokens, 'class': sentiment})
                if sentiment:
                    i += 1
                else:
                    j += 1

            if i == 4000 and j == 4000:
                break

        pickle.dump(reviews, pickle_file, protocol=-1)

with open('TA_restaurants_curated.csv', 'r', encoding="latin1") as trip_advisor_reviews:
    avg = 0
    count = 0
    for row in csv.reader(trip_advisor_reviews):
        if row[0] == '':
            continue
        avg += float(3.0 if row[5] == '' else row[5])
        count += 1
    avg /= count

print('Average rating for trip_advisor reviews: ' + str(avg))

with open('TA_restaurants_curated.csv', 'r', encoding="latin1") as trip_advisor_reviews:
    with open('trip_advisor_data.pickle', 'wb') as pickle_file:
        reviews = list(dict())
        i = 0
        j = 0
        for row in csv.reader(trip_advisor_reviews):
            if row[0] == '':
                continue
            tokens = (preprocess(row[8]))[:-6]
            sentiment = float(3.0 if row[5] == '' else row[5]) >= (avg - 0.5)

            if (i < 4000 and sentiment) or (j < 4000 and not sentiment):
                reviews.append({'tokens': tokens, 'class': sentiment})
                if sentiment:
                    i += 1
                else:
                    j += 1

            if i == 4000 and j == 4000:
                break

        pickle.dump(reviews, pickle_file, protocol=-1)

with open('amazon_data.pickle', 'rb') as amazon_data:
    amazon_reviews = pickle.load(amazon_data)

print("Number of Amazon reviews taken: " + str(len(amazon_reviews)))
# print("processed data:")
# print(amazon_reviews)


with open('trip_advisor_data.pickle', 'rb') as trip_advisor_data:
    trip_advisor_reviews = pickle.load(trip_advisor_data)

print("Number of Trip Advisor reviews taken: " + str(len(trip_advisor_reviews)))
#print("processed data:")
#print(trip_advisor_reviews)