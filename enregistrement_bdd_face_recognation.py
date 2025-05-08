# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:03:38 2025

@author: anais
"""

import cv2
import face_recognition
import os
import pickle

# Dossier contenant les images des visages à encoder
faces_folder = "faces/"
known_faces = {}

# Parcourt chaque image dans le dossier face
for filename in os.listdir(faces_folder):
    # Vérifie si le fichier est une image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(faces_folder, filename)
        image = cv2.imread(image_path)

        # Convertit l'image en RGB 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Génère l'encodage du visage
        encodings = face_recognition.face_encodings(image_rgb)

        # Enregistre le premier encodage 
        if encodings:
            name = os.path.splitext(filename)[0]
            known_faces[name] = encodings[0]
            print(f"Encodage enregistré pour {name}")

# Encodages sauvegarder dans un fichier pickle
with open("faces_encodings.pickle", "wb") as f:
    pickle.dump(known_faces, f)

print("Base de données des visages sauvegardée.")

