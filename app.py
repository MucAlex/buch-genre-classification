from flask import Flask, render_template, request
import pandas as pd
import joblib as job
from src.cleaning import TextProcessor

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    pipeline = job.load(r'.\model.pkl')
    mlb = job.load(r'.\mlb_object.pkl')
    text = request.form['book_text']
    author = request.form['book_author']
    data = pd.DataFrame({'body': text, 'authors': author}, index=[0])
    tp = TextProcessor(text_data=data, text_column='body')
    data['clean_body'] = tp.preprocess(tp.text_data)
    tp = TextProcessor(text_data=data, text_column='authors')
    data['clean_authors'] = tp.clean_text(tp.text_data, for_embedding=False)
    my_prediction = pipeline.predict(data[['clean_body', 'clean_authors']])
    # if author is missing, my prediction contains only zeros
    if my_prediction.any():
        predicted_classes = mlb.classes_[(my_prediction == 1).flatten()]
        predicted_classes = ' '.join(predicted_classes)
    else:
        predicted_classes = 'nicht möglich ohne die Angabe eines Autors :( '

    return render_template('result.html', prediction=predicted_classes)


if __name__ == '__main__':
    app.run(debug=True)
