# -*- coding: utf-8 -*-
"""
Interface graphique pour reconnaissance faciale avec libération de médicament
@author: Anaïs
"""

import cv2
import face_recognition
import pickle
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, window):
        # Initialisation de la fenêtre principale
        self.window = window
        self.window.title("Reconnaissance Faciale - Pilulier")
        self.window.geometry("800x650")

        # Charger les visages connus
        with open("faces_encodings.pickle", "rb") as f:
            self.known_faces = pickle.load(f)

        # Labels pour la vidéo, l'heure et le nom
        self.video_label = tk.Label(self.window)
        self.video_label.pack()

        self.clock_label = tk.Label(self.window, font=("Arial", 16), fg="blue")
        self.clock_label.pack(pady=5)

        self.name_label = tk.Label(self.window, text="Aucun visage détecté", font=("Arial", 16), fg="black")
        self.name_label.pack(pady=5)

        # Initialisation la caméra
        self.cap = cv2.VideoCapture(0)

        # Flag de contrôle
        self.running = False

        # Boutons
        self.start_button = tk.Button(self.window, text="Démarrer la reconnaissance", command=self.start_recognition, bg="pink", fg="white", font=("Arial", 14))
        self.start_button.pack(pady=10)

        self.quit_button = tk.Button(self.window, text="Quitter", command=self.stop_recognition, bg="purple", fg="white", font=("Arial", 14))
        self.quit_button.pack(pady=5)

        # Lancement de la mise à jour de l'heure
        self.update_time()

    def update_time(self):
        
        now = datetime.now()
        current_time = now.strftime("%A %d %B %Y - %H:%M:%S")
        self.clock_label.config(text=current_time)
        self.window.after(1000, self.update_time)

    def show_frame(self):
        
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name_displayed = "Aucun visage détecté"

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(list(self.known_faces.values()), face_encoding)
            name = "Inconnu"

            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_faces.keys())[first_match_index]
            name_displayed = name

            # Dessiner rectangle et nom sur la vidéo
            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(rgb_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Afficher le nom dans l'interface
        self.name_label.config(text=f"Bonjour  {name_displayed}")

        # Affichage dans l'interface
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.window.after(10, self.show_frame)

    def start_recognition(self):
        
        self.running = True
        self.show_frame()

    def stop_recognition(self):
       
        self.running = False
        self.cap.release()
        self.window.destroy()

# Lancer l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
