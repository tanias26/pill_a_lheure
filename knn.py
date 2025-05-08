# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 19:20:51 2025

@author: anais
"""

import os
import csv
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import joblib
 
if os.path.exists("reconnaissance_model.pkl"):
    os.remove("reconnaissance_model.pkl")
    print("Modèle supprimé pour réentraînement.")
 
 
CSV_FILE = "base_reconnaissance.csv"
MODEL_FILE = "reconnaissance_model.pkl"
DIST_THRESHOLD = 0.4201  # Seuil pour la reconnaissance
 
# Initialisation de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
 
# Fonction de normalisation des coordonnées
def normalize_coordinates(coords):
    return coords / np.linalg.norm(coords)  # Normalisation L2
 
# Chargement des données depuis le fichier CSV
def load_data(csv_file):
    X, y = [], []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Sauter l'en-tête
        for row in reader:
            name = row[0]
            coords = list(map(int, row[1:]))
            X.append(normalize_coordinates(np.array(coords)))
            y.append(name)
    return np.array(X), np.array(y)
 
# Entraînement du modèle de reconnaissance faciale
X, y = load_data(CSV_FILE)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X, y)
 
# Sauvegarde du modèle
joblib.dump(knn, MODEL_FILE)
print(f"Modèle de reconnaissance faciale sauvegardé dans {MODEL_FILE}")
 
# Fonction pour effectuer la reconnaissance faciale sur une nouvelle image
def recognize_face(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
 
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        coords = []
 
        # Extraction des points (x, y)
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            coords.extend([x, y])
 
        coords = np.array(coords).reshape(1, -1)
        coords = normalize_coordinates(coords)
 
        # dessin des points de repère du visage
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
 
        # Prédiction du nom de la personne et calcul de la distance
        dist, _ = model.kneighbors(coords)
        predicted_name = model.predict(coords)[0]
 
        print(f"Distance : {dist[0][0]}")  # Affichage la distance pour vérification
 
        if dist[0][0] > DIST_THRESHOLD:
            print("Inconnu (distance trop élevée)")
            return "Inconnu"
        else:
            print(f"Personne reconnue : {predicted_name}")
            return predicted_name
    else:
        print("Aucun visage détecté")
        return "Inconnu"
 
# chargement le modèle de reconnaissance
knn_model = joblib.load(MODEL_FILE)
 
# Ouveture la webcam
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra.")
    exit()
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Échec de la lecture de la vidéo.")
        break
 
    # Redimensionement pour améliorer la vitesse de traitement
    frame_resized = cv2.resize(frame, (640, 480))
 
    # Reconnaissance la personne dans l'image capturée
    recognized_person = recognize_face(frame_resized, knn_model)
 
    # Affichage le cadre avec les résultats de la reconnaissance
    cv2.putText(frame, f"Personne: {recognized_person if recognized_person else 'Inconnu'}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Affichage de l'image avec les points de repère dessinés
    cv2.imshow("Reconnaissance faciale", frame)
 
    # Quitter lorsque la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()