# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:46:43 2025

@author: rache
"""

import cv2

# Charger le modèle pré-entraîné
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger une image
image = cv2.imread('Charles-III.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détecter les visages
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dessiner des rectangles autour des visages détectés
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Afficher le résultat
cv2.imshow('Detection de visage', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
