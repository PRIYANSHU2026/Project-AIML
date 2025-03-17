import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Build and load the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Load weights
model.load_weights("accidents.keras")


class AccidentDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Accident Detection")
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QVBoxLayout()

        # Image display label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        layout.addWidget(self.image_label)

        # Prediction label
        self.prediction_label = QLabel("Prediction: ", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        # Load Image Button
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        # Predict Button
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)

        # Container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Image file path
        self.image_path = None

    def load_image(self):
        # Open a file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.prediction_label.setText("Prediction: ")

    def predict_image(self):
        if not self.image_path:
            self.prediction_label.setText("Prediction: Please load an image first!")
            return

        # Load and preprocess the image
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0  # Normalize

        # Predict
        prediction = model.predict(np.expand_dims(image, axis=0))[0][0]
        label = "Accident" if prediction >= 0.5 else "Not Accident"
        self.prediction_label.setText(f"Prediction: {label}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AccidentDetectionApp()
    window.show()
    sys.exit(app.exec_())
