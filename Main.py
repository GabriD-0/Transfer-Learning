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

# # extrai os arquivos
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
num_classes = len(categories) 

print(categories)


def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return img, x


# # remove corrupted images
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


# normalize data
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)


# summary
print("Carregamento concluído de %d imagens de %d categorias" % (len(data), num_classes))
print("Divisão treino/validação/teste: %d, %d, %d" % (len(x_train), len(x_val), len(x_test)))
print("Formato dos dados de treinamento: ", x_train.shape)
print("Formato dos rótulos de treinamento: ", y_train.shape)


images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis=1)
plt.figure(figsize=(16,4))
plt.imshow(concat_image)
plt.show()


# build the network
model = Sequential()
print("Input dimensions: ",x_train.shape[1:])

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=5,
                    validation_data=(x_val, y_val))


# Visualição das perdas e acuracias
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)




vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
vgg.summary()


# make a reference to VGG's input layer
inp = vgg.input

# make a new softmax layer with num_classes neurons
new_classification_layer = Dense(num_classes, activation='softmax')

# connect our new layer to the second to last layer in VGG, and make a reference to it
out = new_classification_layer(vgg.layers[-2].output)

# create a new network between inp and out
model_new = Model(inp, out)



# make all layers untrainable by freezing weights (except for last layer)
for l, layer in enumerate(model_new.layers[:-1]):
    layer.trainable = False

# ensure the last layer is trainable/not frozen
for l, layer in enumerate(model_new.layers[-1:]):
    layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()


history2 = model_new.fit(x_train, y_train, 
                         batch_size=128, 
                         epochs=5, 
                         validation_data=(x_val, y_val))



fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(121)
ax.plot(history.history["val_loss"])
ax.plot(history2.history["val_loss"])
ax.set_title("validation loss")
ax.set_xlabel("epochs")

ax2 = fig.add_subplot(122)
ax2.plot(history.history["val_accuracy"])
ax2.plot(history2.history["val_accuracy"])
ax2.set_title("validation accuracy")
ax2.set_xlabel("epochs")
ax2.set_ylim(0, 1)

plt.show()



loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)



img, x = get_image('./Finn/1.jpg')
probabilities = model_new.predict([x])