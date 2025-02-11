import os
import numpy as np
import tensorflow as tf
import random
import zipfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.imagenet_utils import preprocess_input
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Activation
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D
from keras._tf_keras.keras.models import Model


dataset_path = r'.\data_set.zip'
extract_path = r'.\data_set' 

# extrai os arquivos
# if os.path.exists(dataset_path):
#     print(f"Tamanho do arquivo: {os.path.getsize(dataset_path)} bytes")
# else:
#     print("Arquivo nao encontrado.")

# if zipfile.is_zipfile(dataset_path):
#     print("O arquivo e um ZIP valido. Extraindo os arquivos...")
    
#     with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#     print("Extracao concluida!")
# else:
#     print("Erro: O arquivo nao e um ZIP valido.")


root = extract_path
train_split, val_split = 0.7, 0.15
categories = [x[0] for x in os.walk(root) if x[0]][1:]

print(categories)


def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return img, x


# remove corrupted images
# import os
# from PIL import Image

# def remove_corrupted_images(extract_path):
#     removed_files = 0
#     for root, _, files in os.walk(extract_path):
#         for file in files:
#             file_path = os.path.join(root, file)
            
#             if os.path.splitext(file)[1].lower() not in ['.jpg', '.jpeg', '.png']:
#                 continue

#             try:
#                 with Image.open(file_path) as img:
#                     img.verify()
                    
#             except Exception as e:
#                 print(f"Removendo imagem corrompida: {file_path} ({e})")
#                 os.remove(file_path)
#                 removed_files += 1

#     print(f"\nTotal de imagens corrompidas removidas: {removed_files}")


# remove_corrupted_images(extract_path)



data = []

for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]

    print(f"Categoria: {category}, Total de imagens encontradas: {len(images)}")

    for img_path in images:
        try:
            img, x = get_image(img_path) 
            data.append({'x': np.array(x[0]), 'y': c})
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}") 
            
        
random.shuffle(data)


idx_val = int(train_split * len(data)) 
idx_test = int((train_split + val_split) * len(data)) 
train = data[:idx_val] 
val = data[idx_val:idx_test] 
test = data[idx_test:] 


x_train, y_train = np.array([t["x"] for t in train]), [t["y"] for t in train]
x_val, y_val = np.array([t["x"] for t in val]), [t["y"] for t in val]
x_test, y_test = np.array([t["x"] for t in test]), [t["y"] for t in test]
print(y_test)

