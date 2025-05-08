# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:45:25 2025

@author: anais
"""
import cv2  
import face_recognition
import pickle

# Chargement des encodages des visages connus depuis le fichier pickle
with open("faces_encodings.pickle", "rb") as f:
    known_faces = pickle.load(f)

# Ouverture de la caméra
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra")
    exit()

while True:
    # Capture d'une image depuis la caméra
    ret, frame = cap.read()

    if not ret:
        print("Échec de la capture d'image")
        break

    # Conversion de l'image en RGB pour face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des emplacements et encodages des visages dans l'image
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Parcours des visages détectés
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparaison des encodages avec les visages connus
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)

        name = "Inconnu"  

        # Si un visage est reconnu
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index]

        # Dessin du rectangle autour du visage et affichage du nom
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Affichage de l'image avec les annotations
    cv2.imshow("Reconnaissance Faciale", frame)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la caméra et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()
