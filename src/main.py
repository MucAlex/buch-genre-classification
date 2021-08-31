from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib as job
from data import DataPreparation
from cleaning import TextProcessor
from classification import MlClassificationPipeline


def load_data():
    data = []
    for dataset in ["train", "test"]:
        data_prep = DataPreparation(file_path=rf'..\daten\klappentext_{dataset}.txt',
                                    label_file_path=rf'..\daten\klappentext_{dataset}_label.txt')
        texts = data_prep.df_text
        texts = texts.dropna(subset=['body', 'authors'])
        label = (data_prep.df_label
                 .set_index('isbn')
                 .loc[texts['isbn']])
        data.append((texts, label))
    return data


if __name__ == '__main__':
    train, test = load_data()
    x_train, y_train = train
    x_test, y_test = test
    for dataset in [x_train, x_test]:
        text_pro = TextProcessor(text_data=dataset, text_column='body')
        dataset['clean_body'] = text_pro.preprocess(text_pro.text_data, for_embedding=False)
        text_pro = TextProcessor(text_data=dataset, text_column='title')
        dataset['clean_title'] = text_pro.preprocess(text_pro.text_data, for_embedding=False)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train.apply(lambda x: [tpc for tpc in x.dropna()], axis=1))
    y_test = mlb.transform(y_test.apply(lambda x: [tpc for tpc in x.dropna()], axis=1))
    #, 'clean_title']]
    mcp = MlClassificationPipeline(x_train=x_train[['clean_body', 'authors']], y_train=y_train)
    gs_list = mcp.perform_grid_search()
    # Linear SVC performs best -> @ index 0
    model = gs_list[0].best_estimator_
    predictions = model.predict(x_test[['clean_body', 'authors']])  #, 'clean_title']])
    print(classification_report(y_test, predictions, digits=4))
    mcp.save_pipe(pipeline=model)
    job.dump(mlb, r'..\mlb_object.pkl')


