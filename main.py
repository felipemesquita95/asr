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

    asr_sys.preprocessing_subsys.preprocess_signal()
    asr_sys.training_data, asr_sys.test_data, asr_sys.training_labels, asr_sys.test_labels = asr_sys.feature_adj_subsys.prepare_to_experiment()


if __name__ == "__main__":
    main()
