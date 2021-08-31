import joblib as job
import pandas as pd
from cleaning import TextProcessor
pipeline = job.load(r'..\model.pkl')
mlb = job.load(r'..\mlb_object.pkl')

text = """Die Beliebtheit der Geistheilung als Alternative zur Schulmedizin wächst ständig. Aber warum ist Geistheilung so erfolgreich? Einer der renommiertesten Heiler Europas gibt unmittelbare Einblicke in seine Tätigkeit. Seriös, kompetent und verständlich beantwortet Horst Krohne die grundlegenden Fragen, stellt Diagnose- und Behandlungsmöglichkeiten vor und zeigt Chancen und Grenzen auf. Für alle, die sich umfassend über Geistheilung informieren oder sich selbst behandeln lassen wollen."""
author = "Horst Krohne"
data = pd.DataFrame({'body': text, 'authors': author}, index=[0])
tp = TextProcessor(text_data=data, text_column='body')
data['clean_body'] = tp.preprocess(tp.text_data)
my_prediction = pipeline.predict(data[['clean_body', 'authors']])
predicted_classes = mlb.classes_[my_prediction == 1]