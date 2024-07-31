from dotenv import load_dotenv
import os

load_dotenv()

class PreprocessingSubsystem:
    def __init__(self, audio_path: str, output_path: str, sampling_rate: int, num_speakers: int, num_utterances: int, num_mfccs: int) -> None:
        self.audio_path = audio_path
        self.output_path = output_path
        self.sampling_rate = sampling_rate
        self.num_speakers = num_speakers
        self.num_utterances = num_utterances
        self.num_mfccs = num_mfccs
   
    def extract_features_from_brsd(self, ):
        print("extracting ...")
