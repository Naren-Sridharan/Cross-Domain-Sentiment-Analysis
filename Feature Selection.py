from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from pprint import pprint


def select_features(file):
    """ Select feature words from reviews based on Information Gain and Chi-square Test"""
    with open(file + '_data.pickle', 'rb') as data:
        reviews = pickle.load(data)
        X = np.array([" ".join(review['tokens']) for review in reviews])
        Y = np.array([review['class'] for review in reviews])
        cv = CountVectorizer(max_df=0.95, min_df=2,
                             max_features=10000)
        X_vec = cv.fit_transform(X)
        select_count = 1000
        res = sorted(list(zip(cv.get_feature_names(),
                       mutual_info_classif(X_vec, Y, discrete_features=True)
                       )), key=lambda x: x[1], reverse=True)[0:select_count]
        print("Top " + str(select_count) + " features according to chi square test:")
        pprint(res)
        print(len(res))

        chi_stats, p_vals = chi2(X_vec, Y)

        chi_res = sorted(list(zip(cv.get_feature_names(),
                       chi_stats
                       )), key=lambda x: x[1], reverse=True)[0:select_count]

        print("Top " + str(select_count) + " features according to chi square test:")
        pprint(chi_res)
        print(len(chi_res))

        selected_features = list(set([x[0] for x in res]) & set([x[0] for x in chi_res]))
        print('The selected features are:')
        pprint(selected_features)
        print(str(len(selected_features)) + " features have been selected")

        with open(file + "_selected_features.pickle", 'wb') as selected_features_pickle:
            pickle.dump(selected_features, selected_features_pickle, protocol=-1)


select_features('amazon')
select_features('trip_advisor')