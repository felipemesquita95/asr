from preprocessing_subsystem.preprocessing_subsystem import PreprocessingSubsystem
from feature_adjustment_subsystem.feature_adjustment_subsystem import FeatureAdjustmentSubsystem
from deep_learning_subsystem.deep_learning_subsystem import DeepLearningSubsystem
from automatic_speaker_recognition_system.automatic_speaker_recognition_system import AutomaticSpeakerRecognitionSystem

from dotenv import load_dotenv
import os

load_dotenv()

def main():

    preprocessing_subsys = PreprocessingSubsystem()
    feature_adj_subsys = FeatureAdjustmentSubsystem()
    deep_learning_subsys = DeepLearningSubsystem()
    asr_sys = AutomaticSpeakerRecognitionSystem(preprocessing_subsys, feature_adj_subsys, deep_learning_subsys)

    # Passo 1: Pré-processar os sinais de áudio
    print("\n" + "="*80)
    print("PASSO 1: PRÉ-PROCESSAMENTO DE SINAIS")
    print("="*80)
    asr_sys.preprocessing_subsys.preprocess_signal()

    # Passo 2: Preparar dados para experimento (ajuste e normalização)
    print("\n" + "="*80)
    print("PASSO 2: PREPARAÇÃO DOS DADOS")
    print("="*80)
    asr_sys.training_data, asr_sys.test_data, asr_sys.training_labels, asr_sys.test_labels = asr_sys.feature_adj_subsys.prepare_to_experiment()

    # Passo 3: Treinar o modelo de deep learning
    print("\n" + "="*80)
    print("PASSO 3: TREINAMENTO E AVALIAÇÃO DO MODELO")
    print("="*80)
    asr_sys.deep_learning_subsys.train(
        asr_sys.training_data,
        asr_sys.training_labels,
        asr_sys.test_data,
        asr_sys.test_labels
    )

    # Passo 4: Salvar modelo
    print("\n" + "="*80)
    print("PASSO 4: SALVANDO MODELO")
    print("="*80)
    asr_sys.deep_learning_subsys.save_model()

    print("\n" + "="*80)
    print("PIPELINE COMPLETO FINALIZADO!")
    print("="*80)


if __name__ == "__main__":
    main()
