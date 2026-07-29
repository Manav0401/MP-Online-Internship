import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model

class FaceCNN(nn.Module):

    def __init__(self, num_classes):

        super(FaceCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1))

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256,19)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x


class_names = [
"Alejandro_Toledo","Alvaro_Uribe","Amelie_Mauresmo","Andre_Agassi",
"Angelina_Jolie","Ariel_Sharon","Arnold_Schwarzenegger",
"Atal_Bihari_Vajpayee","Bill_Clinton","Carlos_Menem",
"Colin_Powell","David_Beckham","Donald_Rumsfeld",
"George_Robertson","George_W_Bush","Gerhard_Schroeder",
"Gloria_Macapagal_Arroyo","Gray_Davis","Guillermo_Coria",
"Hamid_Karzai","Hans_Blix","Hugo_Chavez","Igor_Ivanov",
"Jack_Straw","Jacques_Chirac","Jean_Chretien",
"Jennifer_Aniston","Jennifer_Capriati","Jennifer_Lopez",
"Jeremy_Greenstock","Jiang_Zemin","John_Ashcroft",
"John_Negroponte","Jose_Maria_Aznar",
"Juan_Carlos_Ferrero","Junichiro_Koizumi",
"Kofi_Annan","Laura_Bush","Lindsay_Davenport",
"Lleyton_Hewitt","Luiz_Inacio_Lula_da_Silva",
"Mahmoud_Abbas","Megawati_Sukarnoputri",
"Michael_Bloomberg","Naomi_Watts","Nestor_Kirchner",
"Paul_Bremer","Pete_Sampras","Recep_Tayyip_Erdogan",
"Ricardo_Lagos","Roh_Moo-hyun","Rudolph_Giuliani",
"Saddam_Hussein","Serena_Williams",
"Silvio_Berlusconi","Tiger_Woods","Tom_Daschle",
"Tom_Ridge","Tony_Blair","Vicente_Fox",
"Vladimir_Putin","Winona_Ryder"
]

model = FaceCNN(len(class_names))

model.load_state_dict(torch.load("saved_model/face_cnn.pth", map_location=device))

model.to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor()
])

image = Image.open("test/images.jpg").convert("RGB") #copy your file path
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output,dim=1)

    confidence,predicted = torch.max(probabilities,1)

print("Predicted Person :",class_names[predicted.item()])
print("Confidence :",confidence.item()*100,"%")