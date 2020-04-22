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
from sklearn.ensemble import RandomForestClassifier
import joblib


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('comm', engine)
    X = df['message']
    y = df.drop(columns=['message','genre'])
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

    
def build_model():
    # Feature-union pipeline with KNeighbors
    parameter_rf = {'clf__estimator__n_estimators':[10, 20],
                'features__transformer_weights': ({'text_pipeline': 1, 'starting_verb': 0.5},
                                                  {'text_pipeline': 0.5, 'starting_verb': 1})
    }

    pipeline_rf = Pipeline([
        ('features', FeatureUnion([        
            ('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer())])),
            ('staring_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])

    model = GridSearchCV(pipeline_rf, parameter_rf, cv=2)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    df_ypred = pd.DataFrame(y_pred)
    df_ypred.columns = y_test.columns

    import warnings
    warnings.filterwarnings('ignore')

    for item in category_names:
        print(classification_report(y_test[item], df_ypred[item]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    

def main():
    database_filepath = '/home/workspace/models/em_comm.db'
    model_filepath = '/home/workspace/app/em_comm.joblib'
    
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    print('Building model...')
    model = build_model()
        
    print('Training model...')
    model = model.fit(X_train, y_train)
        
    # print('Evaluating model...')
    # evaluate_model(model, X_test, y_test, category_names)
        
    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')



if __name__ == '__main__':
    main()