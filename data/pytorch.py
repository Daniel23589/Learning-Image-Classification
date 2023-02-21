import os
import PIL
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import random

train_folder = r'.\data\seg_train'
test_folder = r'.\data\seg_test'

def build_csv(directory_string, output_csv_name):

    import csv
    directory = directory_string
    class_lst = os.listdir(directory) #devuelve una LISTA que contiene los nombres de las entradas (nombres de carpetas) en el directorio.
    class_lst.sort() #Importante 
    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #crea nombre de columnas
        for class_name in class_lst:
            class_path = os.path.join(directory, class_name) #concatena varios componentes de ruta con exactamente un separador de directorio ('/') excepto el último componente de ruta.  
            file_list = os.listdir(class_path) #Obtener una lista de archivos en la carpeta de clase
            for file_name in file_list:
                file_path = os.path.join(directory, class_name, file_name) #concatenar directorio de carpeta de clase, nombre de clase y nombre de archivo
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #nombre de la ruta del archivo y nombre de la clase en el archivo csv
    return

build_csv(train_folder, 'train.csv')
build_csv(test_folder, 'test.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

class_zip = zip(train_df['class_index'], train_df['class_name'])
my_list = []
for index, name in class_zip:
  tup = tuple((index, name))
  my_list.append(tup)
unique_list = list(set(my_list))
print('Training:')
print(sorted(unique_list))
print()

class_zip = zip(test_df['class_index'], test_df['class_name'])
my_list = []
for index, name in class_zip:
  tup = tuple((index, name))
  my_list.append(tup)
unique_list = list(set(my_list))
print('Testing:')
print(sorted(unique_list))

class_names = list(train_df['class_name'].unique())
['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

class IntelDataset(torch.utils.data.Dataset): # heredar de la clase Datase
    def __init__(self, csv_file, root_dir="", transform=None):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir # directorio raíz de imágenes, deje "" si usa la columna de ruta de la imagen en el método __getitem__
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df) # longitud de retorno (número de filas) del marco de datos
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx, 1]) #usar la columna de ruta de la imagen (index = 1) en el archivo csv
        image = cv2.imread(image_path) # leer imagen por cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convertir de BGR a RGB para matplotlib
        class_name = self.annotation_df.iloc[idx, 2] # use la columna de nombre de clase (index = 2) en el archivo csv
        class_index = self.annotation_df.iloc[idx, 3] # use la columna de índice de clase (index = 3) en el archivo csv
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index
        
# probar clase de conjunto de datos sin transformación:
train_dataset_untransformed = IntelDataset(csv_file='train.csv', root_dir="", transform=None)

#visualizar 10 imágenes aleatorias del conjunto de datos cargado
plt.figure(figsize=(12,6))
plt.suptitle('untransformed_sample_images')
for i in range(10):
    idx = random.randint(0, len(train_dataset_untransformed))
    image, class_name, class_index = train_dataset_untransformed[idx]
    ax=plt.subplot(2,5,i+1) # create an axis
    ax.title.set_text(class_name + '-' + str(class_index)) # create a name of the axis based on the img name
    plt.imshow(image) # show the img

#plt.show()

from torchvision import transforms

# crear una canalización de transformación
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR)
])
#crear conjuntos de datos con transformaciones:
train_dataset = IntelDataset(csv_file='train.csv', root_dir="", transform=image_transform)
test_dataset = IntelDataset(csv_file='test.csv', root_dir="", transform=image_transform)

#visualizar 10 imágenes aleatorias del conjunto de datos cargado transformed train_dataset
plt.figure(figsize=(12, 6))
plt.suptitle('transformed_sample_images')
for i in range(10):
    idx = random.randint(0, len(train_dataset))
    image, class_name, class_index = train_dataset[idx]
    ax=plt.subplot(2,5,i+1) # crear un eje
    ax.title.set_text(class_name + '-' + str(class_index)) # crear un nombre del eje basado en el nombre de img
    # Las matrices de tensores finales serán de la forma (C * H * W), en lugar de la original (H * W * C), 
    # por lo tanto, usar permute para cambiar el orden
    plt.imshow(image.permute(1, 2, 0)) # muestra la imagen
plt.show()