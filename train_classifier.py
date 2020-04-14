import sys

import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP']:
                    return 1
            except:
                pass
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('comm', engine)
    X = df['message']
    y = df.drop(columns=['message'])
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    # Conver to lower case and remove punctuation and numbers.
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    # Remove multiple spaces.
    text = re.sub(r'\s+', ' ', text)
    # Tokenize
    text = word_tokenize(text)
    # Remove stop words.
    tokens = [w for w in text if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # Feature-union pipeline with KNeighbors
    parameter_knn = {  # 'features__text_pipeline__vect__ngram_range':((1,1), (1,2), (2,2)),
        # 'features__text_pipeline__tfidf__norm':['l1','l2'],
        'clf__estimator__n_neighbors': [5, 10, 20],
        'features__transformer_weights': ({'text_pipeline': 1, 'starting_verb': 0.5},
                                          {'text_pipeline': 0.5, 'starting_verb': 1})
    }

    pipeline_knn = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer())])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))

    ])

    model = GridSearchCV(pipeline_knn, parameter_knn, cv=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    df_ypred = pd.DataFrame(y_pred)

    import warnings
    warnings.filterwarnings('ignore')

    for item in category_names:
        print(classification_report(y_test[item], df_ypred[item]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


'''

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model = model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Optimizing model...')
        model = model.best_estimator_.fit(X_train, y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

'''


def main():
    database_filepath = 'em_comm.db'
    model_filepath = 'em_comm_ide.joblib'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model = model.fit(X_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()

