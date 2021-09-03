from typing import Union, Tuple, List
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib as job
from data import DataPreparation
from cleaning import TextProcessor
from classification import MlClassificationPipeline
import numpy as np


def load_data() -> List[tuple]:
    data = []
    for data_set in ["train", "test"]:
        data_prep = DataPreparation(file_path=rf'..\daten\klappentext_{data_set}.txt',
                                    label_file_path=rf'..\daten\klappentext_{data_set}_label.txt')
        texts = data_prep.df_text
        texts = texts.dropna(subset=['body', 'authors'])
        texts = texts.drop_duplicates(subset=['body'], keep='first')
        label = (data_prep.df_label
                 .set_index('isbn')
                 .loc[texts['isbn']])
        data.append((texts, label))
    return data


def train_model(perform_grid_search: bool = False, save: bool = False, n_cores: int = 4) -> Union[
    Tuple[MlClassificationPipeline, MultiLabelBinarizer], None
]:
    # load data
    train, test = load_data()
    x_train, y_train = train
    x_test, y_test = test

    # clean text columns
    for dataset in [x_train, x_test]:
        text_pro = TextProcessor(text_data=dataset, text_column='body')
        dataset['clean_body'] = text_pro.preprocess(text_pro.text_data, for_embedding=False)
        text_pro = TextProcessor(text_data=dataset, text_column='authors')
        dataset['clean_authors'] = text_pro.clean_text(text_pro.text_data, for_embedding=False)

    # binarize labels
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train.apply(lambda x: [tpc for tpc in x.dropna()], axis=1))
    y_test = mlb.transform(y_test.apply(lambda x: [tpc for tpc in x.dropna()], axis=1))

    # train classification pipeline
    mcp = MlClassificationPipeline(x_train=x_train[['clean_body', 'clean_title', 'clean_authors']], y_train=y_train)
    if perform_grid_search:
        gs_list = mcp.perform_grid_search(num_cores=n_cores)
        scores = [gs.best_score_ for gs in gs_list]
        max_idx = np.argmax(scores)
        # Linear SVC performs best: @ index 0
        model = gs_list[max_idx].best_estimator_
    else:
        model = mcp.train_pipeline()

    predictions = model.predict(x_test[['clean_body', 'clean_title', 'clean_authors']])
    print(classification_report(y_test, predictions, digits=4))
    if save:
        mcp.save_pipe(pipeline=model)
        job.dump(mlb, r'..\mlb_object.pkl')
    else:
        return model, mlb


if __name__ == '__main__':
    model, mlb = train_model(perform_grid_search=False, save=True)



