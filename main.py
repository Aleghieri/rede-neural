import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Carregar o modelo VGG16 pré-treinado (sem as camadas de topo)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base para que não sejam treinadas novamente
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas no topo do modelo
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Criar o novo modelo combinando o modelo base e as camadas personalizadas
model = Model(inputs=base_model.input, outputs=output)

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pré-processamento dos dados
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('caminho/do/diretorio/treinamento', target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory('caminho/do/diretorio/teste', target_size=(224, 224), batch_size=32, class_mode='binary')

# Por questões de segurança eu deixei esse caminho, mas é só você colocar aonde está o arquivo do treinamento e o arquivo do teste.


# Treinar o modelo
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Avaliar o desempenho do modelo
loss, accuracy = model.evaluate(test_generator)
print('Loss:', loss)
print('Accuracy:', accuracy)
