import numpy as np
import math
import os
from dotenv import load_dotenv

load_dotenv()

class FeatureAdjustmentSubsystem:
    
    #os.getenv('SAVES_PATH')
    
    def __init__(self) -> None:
        print("Inicializando Subsistema de Ajuste de Features...")

    def get_frames(self, path, npy_file):
        file_path = os.path.join(path, npy_file)
       
        try:
            mfccs = np.load(file_path)
            num_frames = mfccs.shape[1]
            return num_frames
        
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file_path}")
            return 0
        
        except Exception as e:
            print(f"Erro ao ler arquivo {file_path}: {e}")
            return 0

    def adjust_frames(self, num_frames, mfccs_path):
        
        try:
            data = np.load(mfccs_path)
            current_num_frames = data.shape[1]

            if current_num_frames < num_frames:
                replications = math.ceil(num_frames / current_num_frames)
                replicated_columns = np.tile(data, (1, replications))
                replicated_columns = replicated_columns[:, :num_frames]
                adjusted_data = replicated_columns
            
            elif current_num_frames > num_frames:
                adjusted_data = data[:, :num_frames]
            
            else:
                adjusted_data = data

            return adjusted_data

        except Exception as e:
            print(f"Erro ao processar arquivo {mfccs_path}: {e}")
            return None

    def equalize_frames(self, num_frames, mfccs_path, npy_file):
        if not os.path.exists(mfccs_path):
            print(f"Diretório não encontrado: {mfccs_path}")
            return

        file_path = os.path.join(mfccs_path, npy_file)

        try:
            adjusted_data = self.adjust_frames(num_frames, file_path)
            if adjusted_data is not None:
                output_file_path = os.path.join(mfccs_path, f"{npy_file[:-4]}_{num_frames}.npy")
                np.save(output_file_path, adjusted_data)
                print(f"Arquivo salvo em {output_file_path}")
        except Exception as e:
            print(f"Erro ao processar arquivo {file_path}: {e}")

    def standardize(self, mfccs_path, npy_list):
        matrices = []
        for npy in npy_list:
            file_path = os.path.join(mfccs_path, npy)
            try:
                data = np.load(file_path)
                print(data.shape)
                matrices.append(data)
            except FileNotFoundError:
                print(f"Arquivo não encontrado: {file_path}")
            except Exception as e:
                print(f"Erro ao ler arquivo {file_path}: {e}")
                return

        if len(matrices) != len(npy_list):
            print("Erro: Não foi possível ler todos os arquivos.")
            return

        concatenated_data = np.concatenate(matrices, axis=0)
        print(concatenated_data.shape)

        output_file_path = os.path.join(mfccs_path, "coefficients.npy")
        np.save(output_file_path, concatenated_data)
        print(f"Arquivo 'coefficients.npy' salvo em {output_file_path}")

    def organize_data(self):
        training_data = []
        test_data = []
        training_labels = []
        test_labels = []

        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES')) + 1):
                
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}/coefficients.npy')
                try:
                    normalized_mfccs = np.load(mfccs_path)
                    if utterance == 5:
                        test_data.append(normalized_mfccs)
                        test_labels.append(speaker - 1)
                    else:
                        training_data.append(normalized_mfccs)
                        training_labels.append(speaker - 1)
                except FileNotFoundError:
                    print(f"Arquivo não encontrado: {mfccs_path}")
                except Exception as e:
                    print(f"Erro ao processar arquivo {mfccs_path}: {e}")

        training_data = np.array(training_data)
        test_data = np.array(test_data)
        training_labels = np.array(training_labels)
        test_labels = np.array(test_labels)

        return training_data, test_data, training_labels, test_labels

    def prepare_to_experiment(self):
        max_frames=0

        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES'))+ 1): 
            
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
        
                if self.get_frames(mfccs_path, 'mfccs.npy') > max_frames:
                    num_frames = self.get_frames(mfccs_path, 'mfccs.npy')
                    max_frames = num_frames
        
        print(f"O número de frames será ajustado para {max_frames} frames.")

        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES'))+ 1): 
                
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
                self.equalize_frames(max_frames, mfccs_path, 'mfccs.npy')
                #self.equalize_frames(max_frames, mfccs_path, 'delta.npy')
                #self.equalize_frames(max_frames, mfccs_path, 'deltaDelta.npy')
        
        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES'))+ 1): 
                
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
                npy_list = [f'mfccs_{max_frames}.npy']
                #npy_list = [f'mfccs_{max_frames}.npy', f'delta_{max_frames}.npy', f'deltaDelta_{max_frames}.npy']
                self.standardize(mfccs_path, npy_list)
        
        training_data = []
        test_data = []
        training_labels = []
        test_labels = []

        training_data, test_data, training_labels, test_labels = self.organize_data()
        
        mean = np.mean(training_data, axis=(0, 2), keepdims=True)
        std = np.std(training_data, axis=(0, 2), keepdims=True)

        print(mean.shape, std.shape)

        training_data = (training_data - mean) / std
        test_data = (test_data - mean) / std

        print(training_data.shape, training_labels.shape, test_data.shape, test_labels.shape)
    
        return training_data, test_data, training_labels, test_labels
