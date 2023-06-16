import numpy as np
import tensorflow as tf
import librosa
import json
from pymongo import MongoClient
from flask import Flask

app = Flask(__name__)

# MongoDB connection details
#username_db = "e-stetho"
#password_db = "e-stetho"
mongodb_url ="mongodb+srv://Daoud:BAPft9yNGn9PGrmY@stetho.hyltfoz.mongodb.net/Stetho?retryWrites=true&w=majority"
database_name = 'Stetho'
collection_name = 'sound'

# Connect to MongoDB
client = MongoClient(mongodb_url)
db = client[database_name]
collection = db[collection_name]

# Charger votre modèle TensorFlow ici
model = tf.keras.models.load_model('my_model.h5')


def detect_anomaly(audio_data):
    # Convertir les données audio en nombres flottants
    audio_data_float = np.array(audio_data, dtype=np.float32) / np.iinfo(np.int16).max

    # Fixer la taille des données audio à 3.5 secondes (en fonction du taux d'échantillonnage)
    desired_duration = 3.5  # Durée souhaitée en secondes
    sample_rate = 22050  # Taux d'échantillonnage
    desired_length = int(desired_duration * sample_rate)

    # Vérifier si les données audio doivent être tronquées ou étendues
    if len(audio_data_float) < desired_length:
        if len(audio_data_float) < sample_rate * 3.5:
            return {'error': 'La durée de l\'audio est inférieure à 3.5 secondes.'}
        # Étendre les données audio en ajoutant des zéros à la fin
        audio_data_fixed = np.pad(audio_data_float, (0, desired_length - len(audio_data_float)), mode='constant')
    elif len(audio_data_float) > desired_length:
        # Tronquer les données audio à la longueur désirée
        audio_data_fixed = audio_data_float[:desired_length]
    else:
        audio_data_fixed = audio_data_float

    # Extraire les caractéristiques MFCC
    mfccs = librosa.feature.mfcc(y=audio_data_fixed, sr=sample_rate, n_mfcc=20, n_fft=2048, hop_length=512)

    # Redimensionner les caractéristiques MFCC pour correspondre à la forme d'entrée du modèle
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)

    # Prédire l'anomalie détectée
    predictions = model.predict(mfccs)
    anomalie_detectee = np.argmax(predictions)
    pourcentage_probabilite = round(100 * np.max(predictions), 2)

    # Définir la liste des noms de classes d'anomalies
    classes = ["Artifact", "Extrahls", "Extrastole", "Murmur", "Normal"]

    return {
        'anomalie_detectee': classes[anomalie_detectee],
        'pourcentage_probabilite': pourcentage_probabilite
    }


@app.route('/')
def process_audio_data():
    # Récupérer les données audio depuis MongoDB Atlas
    data = collection.find()
    for d in data:
        audio_data = d["samples"]
        result = detect_anomaly(audio_data)
        anomalie_detectee = result['anomalie_detectee']
        pourcentage_probabilite = result['pourcentage_probabilite']

        # Mettre à jour la collection MongoDB avec les résultats de prédiction
        collection.update_one(
            {'_id': d['_id']},  # Filtrer le document par ID
            {
                "$set": {
                    "anomalie_detectee": anomalie_detectee,
                    "pourcentage_probabilite": pourcentage_probabilite
                }
            }
        )
        print("Résultats de prédiction ajoutés à la collection avec succès.")

    return "Toutes les données ont été traitées."


if __name__ == '__main__':
    app.run()
