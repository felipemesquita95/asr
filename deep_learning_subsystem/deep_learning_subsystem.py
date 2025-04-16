from dotenv import load_dotenv
import tensorflow as tf
import os
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

class DeepLearningSubsystem:
    def __init__(self, taxa_aprendizado=0.001, input_shape=(40, 7241), num_classes=80):
        self.modelo = None
        self.taxa_aprendizado = float(os.getenv('LEARNING_RATE'))
        self.input_shape = input_shape
        self.num_classes = num_classes

        gpus = tf.config.list_physical_devices('GPU')
        
        print("Versão do TensorFlow:", tf.__version__)
        print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
        
        if gpus:
            print("GPU detectada!")
            
            try:
                # Limita o uso de memória da GPU, a memória será alocada conforme necessário
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
               print(e)
        else:
            print("Nenhuma GPU detectada.")
        
    def criar_modelo(self):
        modelo = Sequential()

        modelo.add(Conv1D(64, 4, activation='relu', kernel_regularizer=l2(0.05), input_shape=self.input_shape))
        modelo.add(Flatten())
        modelo.add(Dense(256, activation='relu', kernel_regularizer=l2(0.05)))
        modelo.add(Dense(self.num_classes, activation='softmax'))

        modelo.summary()
        opt = Adam(learning_rate=self.taxa_aprendizado)
        modelo.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.modelo = modelo

        return

    def treinar_modelo(self, dados_treinamento, rotulos_treinamento, dados_teste, rotulos_teste, epochs=100, batch_size=32):
        if self.modelo is None:
            raise ValueError("Modelo não criado. Use o método 'criar_modelo' antes de treinar.")

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=30, min_lr=0.000001)

        historico = self.modelo.fit(
            dados_treinamento, rotulos_treinamento,
            epochs=epochs, batch_size=batch_size,
            validation_data=(dados_teste, rotulos_teste),
            callbacks=[reduce_lr, early_stopping]
        )
        return historico

    def avaliar_modelo(self, dados_teste, rotulos_teste, historico):
        if self.modelo is None:
            raise ValueError("Modelo não criado. Use o método 'criar_modelo' antes de avaliar.")

        previsoes_teste = self.modelo.predict(dados_teste)
        rotulos_preditos = np.argmax(previsoes_teste, axis=1)

        precisao_media = precision_score(rotulos_teste, rotulos_preditos, average='macro')
        revocacao_media = recall_score(rotulos_teste, rotulos_preditos, average='macro')
        f1score_media = f1_score(rotulos_teste, rotulos_preditos, average='macro')

        print("Precisão média:", precisao_media)
        print("Revocação média:", revocacao_media)
        print("F1-score médio:", f1score_media)

        # Exemplo de plot para histórico (opcional)
        training_loss = historico.history['loss']
        validation_loss = historico.history['val_loss']
        plt.figure(figsize=(12, 6))
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.show()
    
     # deep_learning_subsystem.py
    def conjurar_modelo(self, training_data, training_labels, test_data, test_labels):
        # Cria o modelo antes de treinar
        self.criar_modelo()
        
        # Treina o modelo
        self.historico = self.treinar_modelo(training_data, training_labels, test_data, test_labels)
        
        # Avalia o modelo
        self.avaliar_modelo(test_data, test_labels, self.historico) 