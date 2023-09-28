from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
app = Flask(__name__)
model = load_model("DNN.h5")  # Assurez-vous d'avoir le modèle dans le répertoire actuel ou spécifiez le chemin complet
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email_text = data['email_text']

        # Prétraitez le texte de l'e-mail de la même manière que lors de l'entraînement
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(X_train)  # X_train doit correspondre aux données d'entraînement que vous avez utilisées pour former votre modèle DNN


        # Effectuez la prédiction avec le modèle
        prediction = model.predict(tfidf_matrix)
        is_spam = prediction[0][0] > 0.5  # Si la valeur prédite est supérieure à 0.5, c'est considéré comme du spam

        return jsonify({'is_spam': is_spam})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)