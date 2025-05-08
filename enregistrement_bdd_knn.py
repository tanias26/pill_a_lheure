# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:11:28 2025

@author: anais
"""

import os
import csv
import cv2
import mediapipe as mp

# Répertoire contenant les images
BASE_IMAGE_DIR = "face_test"  
# Fichier CSV de sortie
CSV_FILE = "base_reconnaissance.csv"

# Initialisation de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

# Création du fichier CSV et écriture de l'en-tête
with open(CSV_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Création de l'en-tête : "name" suivi des coordonnées x et y pour chaque point
    header = ["name"]
    for i in range(1, 469):  
        header.append(f"x{i}")
        header.append(f"y{i}")
    writer.writerow(header)

    # Parcourt de chaque dossier représentant une personne
    for person_name in os.listdir(BASE_IMAGE_DIR):
        person_folder = os.path.join(BASE_IMAGE_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"\nTraitement de la personne : {person_name}")

        # Parcourt de chaque image du dossier
        for filename in os.listdir(person_folder):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            image_path = os.path.join(person_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Image illisible : {filename}")
                continue

            # Conversion en RGB pour MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

           
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w, _ = image.shape
                coords = []

                # Extraction des coordonnées des landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    coords.extend([x, y])

                # Écriture des coordonnées dans le CSV
                writer.writerow([person_name] + coords)

                # Dessin des landmarks sur l'image
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                # Affichage temporaire de l'image avec landmarks
                cv2.imshow("Landmarks détectés", image)
                cv2.waitKey(500) 
                print(f"Image traitée : {filename} — {len(coords)//2} points détectés")
            else:
                print(f"Aucun visage détecté dans : {filename}")

# Fermeture de MediaPipe et des fenêtres OpenCV
face_mesh.close()
cv2.destroyAllWindows()

print("\nFichier CSV généré :", CSV_FILE)
