from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib as job
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(action='ignore')


class MlClassificationPipeline:

    def __init__(self, x_train, y_train):
        self.y_train = y_train
        self.x_train = x_train

    def train_pipeline(self) -> Pipeline:
        text_processor = ColumnTransformer(transformers=[
            ('body_vect', TfidfVectorizer(min_df=2, max_df=0.6, analyzer='word', ngram_range=(1, 1)), 'clean_body'),
            ('author_vect', CountVectorizer(analyzer='word'), 'clean_authors')
        ])
        svm_pipe = Pipeline([('preprocessor', text_processor),
                             ('cls', MultiOutputClassifier(LinearSVC()))])
        svm_pipe.fit(self.x_train, self.y_train)
        preds = svm_pipe.predict(self.x_train)
        acc = accuracy_score(self.y_train, preds)
        print(f"Training accuracy: {acc:.2%}")
        return svm_pipe

    def perform_grid_search(self, num_cores: int = 4) -> list:
        text_processor = ColumnTransformer(transformers=[
            ('body_vect', TfidfVectorizer(), 'clean_body'),
            ('author_vect', CountVectorizer(), 'clean_authors')
        ])

        svm_pipe = Pipeline([('preprocessor', text_processor),
                             ('cls', MultiOutputClassifier(LinearSVC()))])
        rf_pipe = Pipeline([('preprocessor', text_processor),
                            ('cls', MultiOutputClassifier(RandomForestClassifier(random_state=42)))])
        logr_pipe = Pipeline([('preprocessor', text_processor),
                              ('cls', MultiOutputClassifier(LogisticRegression()))])
        xgb_pipe = Pipeline([('preprocessor', text_processor),
                             ('cls', MultiOutputClassifier(XGBClassifier(random_state=42)))])
        model_dict = {'Linear SVM': svm_pipe,
                      'Random Forest': rf_pipe,
                      'Logistische Regression': logr_pipe,
                      'XGBoost Classifier': xgb_pipe
                      }
        param_grid = {'preprocessor__body_vect__max_df': (0.6,),
                      'preprocessor__body_vect__min_df': (2,),
                      'preprocessor__body_vect__analyzer': ('word',),
                      'preprocessor__author_vect__analyzer': ('word',),
                      'preprocessor__body_vect__ngram_range': ((1, 1),)}
        grid_search = []
        for model, pipe in model_dict.items():
            gs = GridSearchCV(estimator=pipe,
                              param_grid=param_grid,
                              scoring='accuracy',
                              n_jobs=num_cores,
                              verbose=1)
            gs.fit(self.x_train, self.y_train)
            print(f"The best score of {model}: {gs.best_score_:.2%}")
            grid_search.append(gs)
        return grid_search

    @staticmethod
    def save_pipe(pipeline: Pipeline):
        job.dump(pipeline, r'..\model.pkl')

