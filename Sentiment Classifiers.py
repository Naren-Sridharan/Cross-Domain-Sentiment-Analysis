import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np


def accuracy(results, test_y):
    """ Compares results of classifier with targets and gives accuracy"""
    tp, tn, fp, fn, i = 0, 0, 0, 0, 0
    for test in results:
        if test == test_y[i]:
            if test:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if test:
                fp = fp + 1
            else:
                fn = fn + 1
        i = i + 1
    return ((tp + tn) / len(results)) * 100


def classify(model, train_x, train_y, test_x):
    """ Creates model which trains on train_x, train_y and returns predictions for test_x"""
    clf = None

    if model == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
    elif model == 'tree':
        clf = tree.DecisionTreeClassifier()
    elif model == 'svm':
        clf = svm.SVC()
    elif model == 'gauss':
        clf = GaussianNB()
    elif model == 'bernoulli':
        clf = BernoulliNB()
    elif model == 'multinomial':
        clf = MultinomialNB()

    clf.fit(train_x.toarray(), train_y)
    return clf.predict(test_x.toarray())

#loading training data from Amazon
with open('amazon_data.pickle', 'rb') as amazon_data:
    amazon_reviews = pickle.load(amazon_data)

# print(amazon_reviews)
amazon_x = np.array([" ".join(review['tokens']) for review in amazon_reviews])
amazon_y = np.array([review['class'] for review in amazon_reviews])

#training with top features of amazon reviews and testing on trip advisor reviews
with open('amazon_selected_features.pickle', 'rb') as top_features_data:
    top_features = pickle.load(top_features_data)

# print(top_features)

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=top_features)

amazon_x_vec = cv.fit_transform(amazon_x)
# print(amazon_x_vec.toarray())

with open('trip_advisor_data.pickle', 'rb') as trip_advisor_data:
    trip_advisor_reviews = pickle.load(trip_advisor_data)

# print(trip_advisor_reviews)

trip_advisor_x = np.array([" ".join(review['tokens']) for review in trip_advisor_reviews])
trip_advisor_y = np.array([review['class'] for review in trip_advisor_reviews])

trip_advisor_x_vec = cv.fit_transform(trip_advisor_x)

# print(trip_advisor_y)

print('Training on Amazon reviews and testing on Trip Advisor reviews')
print("KNN Accuracy: ", accuracy(classify('knn', amazon_x_vec, amazon_y, trip_advisor_x_vec), trip_advisor_y))
print("Decision Tree Accuracy: ",
      accuracy(classify('tree', amazon_x_vec, amazon_y, trip_advisor_x_vec), trip_advisor_y))
print("SVM Accuracy: ", accuracy(classify('svm', amazon_x_vec, amazon_y, trip_advisor_x_vec), trip_advisor_y))
print("Bernoulli NB Accuracy: ",
      accuracy(classify('bernoulli', amazon_x_vec, amazon_y, trip_advisor_x_vec), trip_advisor_y))
print("Multinomial NB Accuracy: ",
      accuracy(classify('multinomial', amazon_x_vec, amazon_y, trip_advisor_x_vec), trip_advisor_y))


with open('trip_advisor_selected_features.pickle', 'rb') as top_features_data:
    top_features = pickle.load(top_features_data)

# print(top_features)

cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, vocabulary=top_features)

amazon_x_vec = cv.fit_transform(amazon_x)
trip_advisor_x_vec = cv.fit_transform(trip_advisor_x)

print('\nTraining on Trip Advisor reviews and testing on Amazon reviews')
print("KNN Accuracy: ", accuracy(classify('knn', trip_advisor_x_vec, trip_advisor_y, amazon_x_vec), amazon_y))
print("Decision Tree Accuracy: ",
      accuracy(classify('tree', trip_advisor_x_vec, trip_advisor_y, trip_advisor_x_vec), amazon_y))
print("SVM Accuracy: ", accuracy(classify('svm', trip_advisor_x_vec, trip_advisor_y, amazon_x_vec), amazon_y))
print("Bernoulli NB Accuracy: ",
      accuracy(classify('bernoulli', trip_advisor_x_vec, trip_advisor_y, amazon_x_vec), amazon_y))
print("Multinomial NB Accuracy: ",
      accuracy(classify('multinomial', trip_advisor_x_vec, trip_advisor_y, amazon_x_vec), amazon_y))
