from preprocessing_subsystem.preprocessing_subsystem import PreprocessingSubsystem
from feature_adjustment_subsystem.feature_adjustment_subsystem import FeatureAdjustmentSubsystem
from deep_learning_subsystem.deep_learning_subsystem import DeepLearningSubsystem
from automatic_speaker_recognition_system.automatic_speaker_recognition_system import AutomaticSpeakerRecognitionSystem

from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # Instanciando os subsistemas
    preprocessing_subsys = PreprocessingSubsystem()
    feature_adj_subsys = FeatureAdjustmentSubsystem()
    deep_learning_subsys = DeepLearningSubsystem()
    
    # Criando o sistema de reconhecimento de locutor
    asr = AutomaticSpeakerRecognitionSystem(preprocessing_subsys, feature_adj_subsys, deep_learning_subsys)

    # Preparando os dados para o experimento
    asr.training_data, asr.test_data, asr.training_labels, asr.test_labels = asr.feature_adj_subsys.prepare_to_experiment()

    # Treinando e avaliando o modelo
    asr.deep_learning_subsys.conjurar_modelo(asr.training_data, asr.training_labels, asr.test_data, asr.test_labels)
    
if __name__ == "__main__":
    main()

