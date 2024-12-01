import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the ResNet9 model
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        def conv_block(in_channels, out_channels, pool=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512 * 4 * 4, num_classes))  # Adjusted linear layer input size

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        return x

class PlantDiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Load the saved model with map_location to CPU
        self.model = torch.load('/Users/shikarichacha/Desktop/contributo 2/GSOC/Plant diseases detection/plant-disease-model-complete.pth', map_location=torch.device('cpu'))
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjusted image size to 256x256
            transforms.ToTensor(),
        ])

        # Dictionary to map class indices to disease labels
        self.class_names = {
            0: 'Tomato___Late_blight',
            1: 'Tomato___healthy',
            2: 'Grape___healthy',
            3: 'Orange___Haunglongbing_(Citrus_greening)',
            4: 'Soybean___healthy',
            5: 'Squash___Powdery_mildew',
            6: 'Potato___healthy',
            7: 'Corn_(maize)___Northern_Leaf_Blight',
            8: 'Tomato___Early_blight',
            9: 'Tomato___Septoria_leaf_spot',
            10: 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
            11: 'Strawberry___Leaf_scorch',
            12: 'Peach___healthy',
            13: 'Apple___Apple_scab',
            14: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            15: 'Tomato___Bacterial_spot',
            16: 'Apple___Black_rot',
            17: 'Blueberry___healthy',
            18: 'Cherry_(including_sour)___Powdery_mildew',
            19: 'Peach___Bacterial_spot',
            20: 'Apple___Cedar_apple_rust',
            21: 'Tomato___Target_Spot',
            22: 'Pepper,_bell___healthy',
            23: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            24: 'Potato___Late_blight',
            25: 'Tomato___Tomato_mosaic_virus',
            26: 'Strawberry___healthy',
            27: 'Apple___healthy',
            28: 'Grape___Black_rot',
            29: 'Potato___Early_blight',
            30: 'Cherry_(including_sour)___healthy',
            31: 'Corn_(maize)___Common_rust_',
            32: 'Grape___Esca_(Black_Measles)',
            33: 'Raspberry___healthy',
            34: 'Tomato___Leaf_Mold',
            35: 'Tomato___Spider_mites_Two-spotted_spider_mite',
            36: 'Pepper,_bell___Bacterial_spot',
            37: 'Corn_(maize)___healthy'
        }

    def initUI(self):
        self.layout = QVBoxLayout()

        self.imageLabel = QLabel("Upload an Image")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.imageLabel)

        self.uploadButton = QPushButton("Upload Image")
        self.uploadButton.clicked.connect(self.loadImage)
        self.layout.addWidget(self.uploadButton)

        self.predictButton = QPushButton("Predict Disease")
        self.predictButton.clicked.connect(self.predictDisease)
        self.layout.addWidget(self.predictButton)

        self.resultLabel = QLabel("")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.resultLabel)

        self.setLayout(self.layout)
        self.setWindowTitle('Plant Disease Predictor')
        self.show()

    def loadImage(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if filePath:
            pixmap = QPixmap(filePath)
            self.imageLabel.setPixmap(pixmap.scaled(500, 500))
            self.imagePath = filePath

    def predictDisease(self):
        if hasattr(self, 'imagePath'):
            image = Image.open(self.imagePath)
            image = self.transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = self.model(image)
                _, predicted = torch.max(output, 1)
                label = predicted.item()
                disease_name = self.class_names[label]

            self.resultLabel.setText(f'Predicted Disease: {disease_name}')
        else:
            self.resultLabel.setText('Please upload an image first.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PlantDiseasePredictor()
    sys.exit(app.exec_())
