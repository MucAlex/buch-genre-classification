# Klassifikation des Buchgenres anhand des Klappentexts

## Modellentwicklung
Das Training des Klassifikationsmodells erfolgt in train.py. Dieses Skript einmal ausführen, um die Modelldatei und
die Multilabelbinarizer-Datei zu erzeugen.

## Verprobung im Webservice
Durch Ausführen des Skript app.py startet der Webservice. Dieser ermöglicht die Vorhersage für einzelne Bücher anhand
des Klappentexts sowie des Autors des Buchs.