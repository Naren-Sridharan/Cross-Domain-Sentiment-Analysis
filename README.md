# Cross-Domain-Sentiment-Analysis
Cross-Domain Sentiment Analysis Employing Different Feature Selection and Classification Techniques

# Procedure to reproduce results
1. Install nltk (preprocessing), scikit-learn(feature selection and classification), pickle (dump and load objects), pprint (pretty print) etc
2. Download all files into same folder
3. Unzip both data files and run "review preprocessing.py" to get or update the _\_data.pickle_ files
4. Run "feature selection.py" to obtain top features of amazon and trip advisor reviews in seperate _\_selected_features.pickle_ files
5. Run "sentiment classifiers.py" to get the different accuracies obtained by training and testing different classifiers

# Preprocessing Steps
1. Read .csv file
2. Find average rating of reviews 
2. Read row
3. Obtain review from review column - tokenize - remove stop words
4. Assign review class 'positive' if rating >= average rating, 'negative' otherwise
5. Add { 'tokens' : tokens, 'class' : class } to list of review-class list
6. Pickle the review-class list

# Feature Selection
1. Load review-class list
2. X = np array of documents - documents created but joining all tokens
3. Y = np array of classes
4. Use CountVectorizer from sklearn and fit the cocabulary for the documents and obtain document vectors
5. Find info-gain and chi-square test scores for the words (features) and take the top 1000 (select_count)
6. Find list of common top features and save them to _\_.selected_features.pickle_

# Classifiers
The following steps are performed with both amazon and trip advisor data as training and test seperately
1. Load top features of training data from pickle file
2. Fit CountVectorizer with vocabulary = top features for both training and test tokens
3. Use classifier to fit on training data and and return predictions
4. Find accuracy with returned predictions vs test targets and return accuracy

# References - Implementation of paper
Ahuja C., Sivasankar E. (2018) Cross-Domain Sentiment Analysis Employing Different Feature Selection and Classification Techniques. In: Mishra D., Nayak M., Joshi A. (eds) Information and Communication Technology for Sustainable Development. Lecture Notes in Networks and Systems, vol 10. Springer, Singapore
