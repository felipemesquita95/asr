from preprocessing_subsystem.preprocessing_subsystem import PreprocessingSubsystem
from feature_adjustment_subsystem.feature_adjustment_subsystem import FeatureAdjustmentSubsystem
from deep_learning_subsystem.deep_learning_subsystem import DeepLearningSubsystem
from automatic_speaker_recognition_system.automatic_speaker_recognition_system import AutomaticSpeakerRecognitionSystem

from dotenv import load_dotenv
import os

load_dotenv()

audio_path = os.getenv('UTTERANCES_PATH')
output_path = os.getenv('SAVES_PATH')
sampling_rate = int(os.getenv('UTTERANCES_SAMPLING_RATE'))
num_speakers = int(os.getenv('NUM_SPEAKERS'))
num_utterances = int(os.getenv('NUM_UTTERANCES'))
num_mfccs = int(os.getenv('NUM_MFCCS'))

def main():
    
    preprocessing_subsys = PreprocessingSubsystem(audio_path, output_path, sampling_rate, num_speakers, num_utterances, num_mfccs)
    feature_adj_subsys = FeatureAdjustmentSubsystem()
    deep_learning_subsys = DeepLearningSubsystem()

    asr_sys = AutomaticSpeakerRecognitionSystem(preprocessing_subsys, feature_adj_subsys, deep_learning_subsys)

       
    asr_sys.preprocessing_subsys.extract_features_from_brsd()


if __name__ == "__main__":
    main()
