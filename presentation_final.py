# -*- coding: utf-8 -*-
"""
Created on Fri May  2 17:40:22 2025

@author: anais
"""
import tkinter as tk
import subprocess

class InterfaceReconnaissanceFaciale:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface Reconnaissance Faciale")
        self.root.geometry("300x200")        
        self.create_widgets()

    def create_widgets(self):
       
        self.btn1 = tk.Button(self.root, text="Face Recognition", command=self.run_face_recognition, height=2, width=25)
        self.btn1.pack(pady=10)

       
        self.btn2 = tk.Button(self.root, text="KNN Recognition", command=self.run_knn_recognition, height=2, width=25)
        self.btn2.pack(pady=10)

       
        self.btn_quit = tk.Button(self.root, text="Quitter", command=self.quit, height=2, width=25)
        self.btn_quit.pack(pady=10)

    def run_face_recognition(self):
        subprocess.Popen(["python", "interface_face_recogation.py"])

    def run_knn_recognition(self):
        subprocess.Popen(["python", "interface_knn.py"])

    def quit(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceReconnaissanceFaciale(root)
    root.mainloop()
