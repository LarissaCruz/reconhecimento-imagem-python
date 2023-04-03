import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from PIL import Image

# Carregar as imagens dos ovos e seus respectivos rótulos
X_train = []
y_train = []
for i in range(2):
    img = cv2.imread(f"ovos/{i}_sujo.jpg", cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (128, 128))
    X_train.append(img_resize)
    y_train.append(0)  # categoria 0: ovo sujo
    diff_threshold = 30 # definir um threshold de diferença entre os pixels
    diff = cv2.absdiff(img_resize, cv2.medianBlur(img_resize, 21)) # calcular a diferença entre os pixels da imagem e sua versão suavizada
    max_diff = np.max(diff) # obter o valor máximo da diferença

for i in range(2):
    img = cv2.imread(f"ovos/{i}_limpo.jpg", cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (128, 128))
    X_train.append(img_resize)
    y_train.append(1)  # categoria 1: ovo limpo
    diff_threshold = 30 # definir um threshold de diferença entre os pixels
    diff = cv2.absdiff(img_resize, cv2.medianBlur(img_resize, 21)) # calcular a diferença entre os pixels da imagem e sua versão suavizada
    max_diff = np.max(diff) # obter o valor máximo da diferença

for i in range(5):
    img = cv2.imread(f"ovos/{i}_trincado.jpg", cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img, (128, 128))
    X_train.append(img_resize)
    y_train.append(2)  # categoria 2: ovo rachado
    diff_threshold = 30 # definir um threshold de diferença entre os pixels
    diff = cv2.absdiff(img_resize, cv2.medianBlur(img_resize, 21)) # calcular a diferença entre os pixels da imagem e sua versão suavizada
    max_diff = np.max(diff) # obter o valor máximo da diferença

# Converter os rótulos para o formato one-hot encoding
y_train = to_categorical(y_train, num_classes=3)

# Converter as imagens para o formato apropriado para a CNN
X_train = np.array(X_train).reshape(-1, 128, 128, 1)
X_train = X_train / 255.0  # normalizar os pixels

# Definir a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

#Testar o modelo com uma nova imagem
img_test = cv2.imread("ovos/0_sujo.jpg",  cv2.IMREAD_GRAYSCALE)
img_test = cv2.resize(img_test, (128, 128))
img_test = img_test.reshape(1, 128, 128, 1) / 255.0
y_pred = model.predict(img_test)
print(y_pred)
blur = cv2.medianBlur(img_test, 21)
diff = cv2.absdiff(img_test.squeeze(), blur)
max_diff = np.max(diff)
print(max_diff)
if np.argmax(y_pred) == 0:
    print("O ovo é sujo")
elif np.argmax(y_pred) == 1:
    print("O ovo é limpo")
else:
    # Verificar se a diferença entre os pixels da imagem e sua versão suavizada é maior que o threshold
   
    if max_diff > 30:
        print("O ovo está trincado")
    else:
        print("O ovo é trincado, mas não ultrapassa o threshold definido")