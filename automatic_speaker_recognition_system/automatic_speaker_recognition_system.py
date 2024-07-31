from preprocessing_subsystem.preprocessing_subsystem import PreprocessingSubsystem
from feature_adjustment_subsystem.feature_adjustment_subsystem import FeatureAdjustmentSubsystem
from deep_learning_subsystem.deep_learning_subsystem import DeepLearningSubsystem

from dotenv import load_dotenv

load_dotenv()

class AutomaticSpeakerRecognitionSystem:
    def __init__(self, 
                 preprocessing_subsys: PreprocessingSubsystem,
                 feature_adjustment_subsystem: FeatureAdjustmentSubsystem,
                 deep_learning_subsystem: DeepLearningSubsystem
                ) -> None:
        
        self.preprocessing_subsys = preprocessing_subsys
        self.feature_adj_subsys = feature_adjustment_subsystem
        self.deep_learning_subsys = deep_learning_subsystem
        
        print("init ...")
