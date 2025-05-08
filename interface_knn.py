# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:10:25 2025

@author: anais
"""

import os
import cv2
import csv
import joblib
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
import datetime

# Fichier CSV contenant les données d'entraînement
CSV_FILE = "base_reconnaissance.csv"
# Fichier pour le modèle KNN
MODEL_FILE = "reconnaissance_model.pkl"
# Seuil de distance pour la reconnaissance
DIST_THRESHOLD = 0.35 

# Si le fichier modèle existe, il est supprimé pour réentraînement
if os.path.exists("reconnaissance_model.pkl"):
    os.remove("reconnaissance_model.pkl")
    print("🗑️ Modèle supprimé pour réentraînement.")


def normalize_coordinates(coords):
   
    return coords / np.linalg.norm(coords)


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

# Si le modèle n'existe pas, il est entraîné
if not os.path.exists(MODEL_FILE):
    X, y = load_data(CSV_FILE)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X, y)
    joblib.dump(knn, MODEL_FILE)

# Chargement du modèle KNN
model = joblib.load(MODEL_FILE)

# Initialisation de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

class FaceRecognitionApp:
    def __init__(self, window):
        
        self.window = window
        self.window.title("Reconnaissance Faciale - Pilulier")
        self.window.geometry("800x700")

        # Label pour la vidéo
        self.video_label = tk.Label(window)
        self.video_label.pack()

        # Label pour afficher le message de reconnaissance
        self.status_label = tk.Label(window, text="", font=("Arial", 16), fg="blue")
        self.status_label.pack(pady=10)

        # Label pour la date et l'heure
        self.datetime_label = tk.Label(window, text="", font=("Arial", 16), fg="blue")
        self.datetime_label.pack(pady=5)

        # Bouton pour démarrer la reconnaissance
        self.start_button = tk.Button(window, text="Démarrer la reconnaissance", command=self.start, bg="pink", fg="white", font=("Arial", 14))
        self.start_button.pack(pady=5)

        # Bouton pour quitter
        self.quit_button = tk.Button(window, text="Quitter", command=self.quit, bg="purple", fg="white", font=("Arial", 14))
        self.quit_button.pack(pady=5)

        # Capture vidéo
        self.cap = cv2.VideoCapture(0)
        self.running = False

        # Lancement de l'actualisation de l'heure
        self.update_datetime()

    def update_datetime(self):
        
        now = datetime.datetime.now()
        formatted = now.strftime("%d/%m/%Y %H:%M:%S")
        self.datetime_label.config(text=f"{formatted}")
        self.window.after(1000, self.update_datetime)

    def start(self):
        
        self.running = True
        self.update_frame()

    def quit(self):
        
        self.running = False
        self.cap.release()
        self.window.destroy()

    def recognize_face(self, frame):
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        name = "Inconnu"
        message = "Aucun visage détecté."

        # Si un visage est détecté
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            coords = []
            xs, ys = [], []

            # Extraction des coordonnées des landmarks
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                xs.append(x)
                ys.append(y)
                coords.extend([x, y])

            # Normalisation des coordonnées
            coords = np.array(coords).reshape(1, -1)
            coords = normalize_coordinates(coords)

            # Rectangle autour du visage
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # Prédiction du visage
            dist, _ = model.kneighbors(coords)
            prediction = model.predict(coords)[0]

            # Si la distance est inférieure au seuil, le visage est reconnu
            if dist[0][0] <= DIST_THRESHOLD:
                name = prediction
                message = f"Bonjour {name}"
            else:
                message = "Visage inconnu"

        self.status_label.config(text=message, fg="black")
        return name

    def update_frame(self):
        
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        self.recognize_face(frame)

        # Conversion en RGB pour l'affichage dans Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Répéter la capture
        self.window.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
