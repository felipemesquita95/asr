from dotenv import load_dotenv
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

class DeepLearningSubsystem:
    def __init__(self) -> None:
        # Parâmetros da arquitetura
        self.conv1d_filters = int(os.getenv('CONV1D_FILTERS'))
        self.conv1d_kernel_size = int(os.getenv('CONV1D_KERNEL_SIZE'))
        self.dense_units = int(os.getenv('DENSE_UNITS'))

        # Parâmetros de regularização
        self.l2_reg = float(os.getenv('L2_REGULARIZATION'))

        # Parâmetros de treinamento
        self.batch_size = int(os.getenv('BATCH_SIZE'))
        self.epochs = int(os.getenv('EPOCHS'))
        self.learning_rate = float(os.getenv('LEARNING_RATE'))

        # Parâmetros de callbacks
        self.early_stopping_patience = int(os.getenv('EARLY_STOPPING_PATIENCE'))
        self.reduce_lr_patience = int(os.getenv('REDUCE_LR_PATIENCE'))
        self.reduce_lr_factor = float(os.getenv('REDUCE_LR_FACTOR'))
        self.reduce_lr_min_lr = float(os.getenv('REDUCE_LR_MIN_LR'))

        # Configurações gerais
        self.num_speakers = int(os.getenv('NUM_SPEAKERS'))
        self.saves_path = os.getenv('SAVES_PATH')

        # Modelo e histórico
        self.model = None
        self.history = None

        print("Inicializando Subsistema de aprendizado profundo...")

    def criarModelo(self, input_shape):
        """
        Constrói o modelo de rede neural para reconhecimento de locutor.

        Arquitetura (baseada no código original):
        - Conv1D: Extrai padrões temporais dos MFCCs
        - Flatten: Transforma em vetor 1D
        - Dense: Camada totalmente conectada com regularização L2
        - Output: Classificação softmax para N locutores
        """
        modelo = keras.Sequential([
            # Conv1D para extrair features temporais
            layers.Conv1D(
                filters=self.conv1d_filters,
                kernel_size=self.conv1d_kernel_size,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                input_shape=input_shape
            ),

            # Flatten para conectar com camada densa
            layers.Flatten(),

            # Camada densa com regularização L2
            layers.Dense(
                self.dense_units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),

            # Camada de saída (classificação multi-classe)
            layers.Dense(self.num_speakers, activation='softmax')
        ])

        # Compilar modelo
        opt = Adam(learning_rate=self.learning_rate)
        modelo.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = modelo
        print("\nArquitetura do modelo:")
        modelo.summary()

        return modelo

    def treinarModelo(self, dadosTreinamento, rotulosTreinamento, dadosTeste, rotulosTeste):
        """
        Treina o modelo de reconhecimento de locutor.

        Args:
            dadosTreinamento: Dados de treino (N, features, frames)
            rotulosTreinamento: Labels dos locutores (N,)
            dadosTeste: Dados de teste
            rotulosTeste: Labels de teste
        """
        if self.model is None:
            input_shape = (dadosTreinamento.shape[1], dadosTreinamento.shape[2])
            self.criarModelo(input_shape)

        print(f"\nIniciando treinamento...")
        print(f"Dados de treino: {dadosTreinamento.shape}")
        print(f"Labels de treino: {rotulosTreinamento.shape}")
        print(f"Dados de teste: {dadosTeste.shape}")
        print(f"Labels de teste: {rotulosTeste.shape}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.reduce_lr_min_lr,
            verbose=1
        )

        # Transpor dados: (N, features, frames) -> (N, frames, features)
        # Conv1D espera (batch, timesteps, features)
        dadosTreinamento_transposed = np.transpose(dadosTreinamento, (0, 2, 1))
        dadosTeste_transposed = np.transpose(dadosTeste, (0, 2, 1))

        # Treinar modelo
        historico = self.model.fit(
            dadosTreinamento_transposed,
            rotulosTreinamento,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(dadosTeste_transposed, rotulosTeste),
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )

        self.history = historico
        print("\nTreinamento concluído!")

        return self.model, historico

    def avaliarModelo(self, dadosTeste, rotulosTeste, historicoEpocas=None):
        """
        Avalia o modelo nos dados de teste.

        Args:
            dadosTeste: Dados de teste (N, features, frames)
            rotulosTeste: Labels verdadeiros (N,)
            historicoEpocas: Histórico de treinamento (opcional)
        """
        if historicoEpocas is None:
            historicoEpocas = self.history

        print("\n" + "="*80)
        print("AVALIAÇÃO DO MODELO")
        print("="*80)

        # Transpor dados
        dadosTeste_transposed = np.transpose(dadosTeste, (0, 2, 1))

        # Fazer previsões com o modelo
        previsoesTeste = self.model.predict(dadosTeste_transposed, verbose=0)
        rotulosPreditos = np.argmax(previsoesTeste, axis=1)

        # Avaliar
        loss, accuracy = self.model.evaluate(dadosTeste_transposed, rotulosTeste, verbose=0)
        print(f"\nLoss: {loss:.4f}")
        print(f"Acurácia: {accuracy*100:.2f}%")

        # Calcular métricas adicionais
        precisaoMedia = precision_score(rotulosTeste, rotulosPreditos, average='macro', zero_division=0)
        revocacaoMedia = recall_score(rotulosTeste, rotulosPreditos, average='macro', zero_division=0)
        f1scoreMedia = f1_score(rotulosTeste, rotulosPreditos, average='macro', zero_division=0)

        print("\nMétricas:")
        print(f"Precisão média: {precisaoMedia:.4f}")
        print(f"Revocação média: {revocacaoMedia:.4f}")
        print(f"F1-score médio: {f1scoreMedia:.4f}")

        # Plotar curvas de loss e accuracy
        if historicoEpocas is not None:
            self.plotar_curvas_treinamento(historicoEpocas)

        # Plotar matriz de confusão
        self.plotar_matriz_confusao(rotulosTeste, rotulosPreditos)

        return accuracy, rotulosPreditos

    def plotar_curvas_treinamento(self, historicoEpocas):
        """
        Plota curvas de Loss e Accuracy do treinamento.
        """
        trainingLoss = historicoEpocas.history['loss']
        validationLoss = historicoEpocas.history['val_loss']
        trainingAccuracy = historicoEpocas.history['accuracy']
        validationAccuracy = historicoEpocas.history['val_accuracy']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(trainingLoss, label='Training Loss', linewidth=2)
        ax1.plot(validationLoss, label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Loss Curves', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(trainingAccuracy, label='Training Accuracy', linewidth=2)
        ax2.plot(validationAccuracy, label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Accuracy Curves', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.saves_path, 'training_curves.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"\nCurvas de treinamento salvas em: {output_path}")
        plt.close()

    def plotar_matriz_confusao(self, rotulosTeste, rotulosPreditos):
        """
        Plota a matriz de confusão.
        """
        # Calcular a matriz de confusão
        cm = confusion_matrix(rotulosTeste, rotulosPreditos)

        # Plotar a matriz de confusão
        plt.figure(figsize=(20, 18))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt='g',
            cbar=True,
            square=True,
            annot_kws={"size": 6}
        )

        # Define manualmente os rótulos dos eixos
        rotulosClasses = [f"Locutor {i}" for i in range(self.num_speakers)]
        plt.xticks(
            ticks=np.arange(self.num_speakers) + 0.5,
            labels=rotulosClasses,
            rotation=90,
            ha='center',
            fontsize=6
        )
        plt.yticks(
            ticks=np.arange(self.num_speakers) + 0.5,
            labels=rotulosClasses,
            rotation=0,
            va='center',
            fontsize=6
        )

        plt.title('Matriz de Confusão', fontsize=14)
        plt.xlabel('Predições', fontsize=12)
        plt.ylabel('Rótulos Verdadeiros', fontsize=12)

        output_path = os.path.join(self.saves_path, 'confusion_matrix.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {output_path}")
        plt.close()

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

    # Aliases para manter compatibilidade
    def train(self, training_data, training_labels, test_data=None, test_labels=None):
        """Alias para treinarModelo mantendo compatibilidade com código anterior"""
        if test_data is None or test_labels is None:
            raise ValueError("Dados de teste são obrigatórios para esta arquitetura")

        modelo, historico = self.treinarModelo(training_data, training_labels, test_data, test_labels)
        self.avaliarModelo(test_data, test_labels, historico)
        return historico

    def evaluate(self, test_data, test_labels):
        """Alias para avaliarModelo mantendo compatibilidade com código anterior"""
        return self.avaliarModelo(test_data, test_labels)
