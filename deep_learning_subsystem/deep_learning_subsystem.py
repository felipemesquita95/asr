from dotenv import load_dotenv
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

load_dotenv()

class DeepLearningSubsystem:
    def __init__(self) -> None:
        # Parâmetros da arquitetura
        self.conv1d_filters = int(os.getenv('CONV1D_FILTERS'))
        self.conv1d_kernel_size = int(os.getenv('CONV1D_KERNEL_SIZE'))
        self.pool_size = int(os.getenv('POOL_SIZE'))
        self.dense_units = int(os.getenv('DENSE_UNITS'))
        self.dropout_rate = float(os.getenv('DROPOUT_RATE'))

        # Parâmetros de regularização
        self.l2_reg = float(os.getenv('L2_REGULARIZATION'))

        # Parâmetros de treinamento
        self.batch_size = int(os.getenv('BATCH_SIZE'))
        self.epochs = int(os.getenv('EPOCHS'))
        self.learning_rate = float(os.getenv('LEARNING_RATE'))
        self.validation_split = float(os.getenv('VALIDATION_SPLIT'))

        # Configurações gerais
        self.num_speakers = int(os.getenv('NUM_SPEAKERS'))
        self.saves_path = os.getenv('SAVES_PATH')

        # Modelo e histórico
        self.model = None
        self.history = None

        print("Inicializando Subsistema de aprendizado profundo...")

    def build_model(self, input_shape):
        """
        Constrói o modelo de rede neural para reconhecimento de locutor.

        Arquitetura:
        - Conv1D: Extrai padrões temporais dos MFCCs
        - MaxPooling1D: Reduz dimensionalidade
        - Flatten: Transforma em vetor 1D
        - Dense: Camada totalmente conectada com regularização
        - Dropout: Previne overfitting
        - Output: Classificação softmax para N locutores
        """
        model = keras.Sequential([
            # Camada de entrada
            layers.Input(shape=input_shape),

            # Conv1D para extrair features temporais
            layers.Conv1D(
                filters=self.conv1d_filters,
                kernel_size=self.conv1d_kernel_size,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                padding='same'
            ),

            # MaxPooling para reduzir dimensionalidade
            layers.MaxPooling1D(pool_size=self.pool_size),

            # Flatten para conectar com camada densa
            layers.Flatten(),

            # Camada densa com dropout e regularização
            layers.Dense(
                self.dense_units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            layers.Dropout(self.dropout_rate),

            # Camada de saída (classificação multi-classe)
            layers.Dense(self.num_speakers, activation='softmax')
        ])

        # Compilar modelo
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("\nArquitetura do modelo:")
        model.summary()

        return model

    def train(self, training_data, training_labels, test_data=None, test_labels=None):
        """
        Treina o modelo de reconhecimento de locutor.

        Args:
            training_data: Dados de treino (N, features, frames)
            training_labels: Labels dos locutores (N,)
            test_data: Dados de teste (opcional)
            test_labels: Labels de teste (opcional)
        """
        if self.model is None:
            input_shape = (training_data.shape[1], training_data.shape[2])
            self.build_model(input_shape)

        print(f"\nIniciando treinamento...")
        print(f"Dados de treino: {training_data.shape}")
        print(f"Labels de treino: {training_labels.shape}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")

        # Callbacks para melhorar o treinamento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Transpor dados: (N, features, frames) -> (N, frames, features)
        # Conv1D espera (batch, timesteps, features)
        training_data_transposed = np.transpose(training_data, (0, 2, 1))

        # Treinar modelo
        self.history = self.model.fit(
            training_data_transposed,
            training_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )

        print("\nTreinamento concluído!")

        # Se houver dados de teste, avaliar
        if test_data is not None and test_labels is not None:
            self.evaluate(test_data, test_labels)

        return self.history

    def evaluate(self, test_data, test_labels):
        """
        Avalia o modelo nos dados de teste.

        Args:
            test_data: Dados de teste (N, features, frames)
            test_labels: Labels verdadeiros (N,)
        """
        print("\n" + "="*60)
        print("AVALIAÇÃO NO CONJUNTO DE TESTE")
        print("="*60)

        # Transpor dados
        test_data_transposed = np.transpose(test_data, (0, 2, 1))

        # Avaliar
        loss, accuracy = self.model.evaluate(test_data_transposed, test_labels, verbose=0)

        print(f"\nLoss: {loss:.4f}")
        print(f"Acurácia: {accuracy*100:.2f}%")

        # Predições
        predictions = self.model.predict(test_data_transposed, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        # Relatório de classificação
        print("\n" + "-"*60)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("-"*60)
        print(classification_report(test_labels, predicted_labels, zero_division=0))

        # Matriz de confusão (resumida para não poluir)
        conf_matrix = confusion_matrix(test_labels, predicted_labels)
        accuracy_per_speaker = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

        print("\n" + "-"*60)
        print("ACURÁCIA POR LOCUTOR")
        print("-"*60)
        print(f"Média: {np.mean(accuracy_per_speaker)*100:.2f}%")
        print(f"Mediana: {np.median(accuracy_per_speaker)*100:.2f}%")
        print(f"Mínima: {np.min(accuracy_per_speaker)*100:.2f}%")
        print(f"Máxima: {np.max(accuracy_per_speaker)*100:.2f}%")

        return accuracy, predicted_labels

    def predict(self, data):
        """
        Faz predições para novos dados.

        Args:
            data: Dados de entrada (N, features, frames)

        Returns:
            predicted_labels: Labels preditos (N,)
            probabilities: Probabilidades para cada classe (N, num_speakers)
        """
        # Transpor dados
        data_transposed = np.transpose(data, (0, 2, 1))

        # Predições
        probabilities = self.model.predict(data_transposed, verbose=0)
        predicted_labels = np.argmax(probabilities, axis=1)

        return predicted_labels, probabilities

    def save_model(self, filepath=None):
        """
        Salva o modelo treinado.

        Args:
            filepath: Caminho para salvar o modelo (opcional)
        """
        if filepath is None:
            filepath = os.path.join(self.saves_path, 'model.keras')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"\nModelo salvo em: {filepath}")

    def load_model(self, filepath=None):
        """
        Carrega um modelo salvo.

        Args:
            filepath: Caminho do modelo salvo (opcional)
        """
        if filepath is None:
            filepath = os.path.join(self.saves_path, 'model.keras')

        self.model = keras.models.load_model(filepath)
        print(f"\nModelo carregado de: {filepath}")

    def plot_training_history(self, output_path=None):
        """
        Plota o histórico de treinamento (loss e accuracy).

        Args:
            output_path: Caminho para salvar o gráfico (opcional)
        """
        if self.history is None:
            print("Nenhum histórico de treinamento disponível.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Treino', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Loss durante o treinamento', fontsize=12)
        ax1.set_xlabel('Época', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Treino', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Validação', linewidth=2)
        ax2.set_title('Acurácia durante o treinamento', fontsize=12)
        ax2.set_xlabel('Época', fontsize=10)
        ax2.set_ylabel('Acurácia', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(self.saves_path, 'training_history.png')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"\nGráfico de treinamento salvo em: {output_path}")
        plt.close()
