import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
import pickle
from sklearn.externals import joblib

train = pd.read_csv('text_emotion_train.csv').fillna(' ')
test = pd.read_csv('text_emotion_test.csv').fillna(' ')

class_names = ['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry', 'depressed']

train_text = train['content']
test_text = test['content']
all_text = pd.concat([train_text, test_text])

#vectorizing - converting word forms to numerical forms by assigning id to every word ( "\w{1,}" word has a min len of 1), making use of the numbers further.
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)

#constructing document-term matrix using transform --read document-term matrix
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

#Todo: read what is char_vectorizer and why we need (other than word vectorizer, char vectorizer helps in taking individual characters into consideration by that adding further features for classification)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=30000)
char_vectorizer.fit(all_text)

#constructing document-term matrix using transform --read document-term matrix
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


#combining both word and char vectorizers horizantally
"""
hstack example
>>> from scipy.sparse import coo_matrix, hstack
>>> A = coo_matrix([[1, 2], [3, 4]])
>>> B = coo_matrix([[5], [6]])
>>> hstack([A,B]).toarray()
array([[1, 2, 5],
       [3, 4, 6]])
"""
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


#training the model
def train_model():
    losses = []
    predictions = {'id': test['id']}
    for class_name in class_names:
        train_target = train[class_name]
        #sag optimizer for L2 regularization. --readegularization(prevents overfitting by adding weights to the features)
        classifier = LogisticRegression(solver='sag')

        #3-fold cross validation
        cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        losses.append(cv_loss)
        print('CV score for class {} is {}'.format(class_name, cv_loss))

        classifier.fit(train_features, train_target)
        # save the model to disk
        filename = 'models/'+class_name+'_model.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

    print('Total CV score is {}'.format(np.mean(losses)))

    submission = pd.DataFrame.from_dict(predictions)
    submission.to_csv('word_submission.csv', index=False)

def predict(text):
    model, prob = [], []
    for filename in class_names:
        loaded_model = joblib.load("models/"+filename+"_model.sav")
        test_word_features = word_vectorizer.transform([text])
        test_char_features = char_vectorizer.transform([text])
        test_features = hstack([test_char_features, test_word_features])
        result = loaded_model.predict_proba(test_features)[:, 1]
        prob.append(result)
        model.append(filename)
    return model, prob


if __name__ == "__main__":
    model, prob = predict(["friday and i am still working"])
    print model[prob.index(max(prob))]
    print model, prob