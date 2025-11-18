"""
Script para executar experimentos completos de validação cruzada
com diferentes configurações de MFCCs e divisões treino/teste.

Experimentos:
- MFCCs: 10, 20, 30, 40 coeficientes
- 1 amostra de teste: 5 experimentos (leave-one-out)
- 2 amostras de teste: 10 experimentos (todas combinações)
- Total: 60 experimentos independentes
"""

from preprocessing_subsystem.preprocessing_subsystem import PreprocessingSubsystem
from feature_adjustment_subsystem.feature_adjustment_subsystem import FeatureAdjustmentSubsystem
from deep_learning_subsystem.deep_learning_subsystem import DeepLearningSubsystem
from automatic_speaker_recognition_system.automatic_speaker_recognition_system import AutomaticSpeakerRecognitionSystem

from dotenv import load_dotenv
import os
import numpy as np
import json
from datetime import datetime
from itertools import combinations

load_dotenv()


# Configurações de experimentos
NUM_MFCC_OPTIONS = [10, 20, 30, 40]

# Experimentos com 1 amostra de teste (leave-one-out)
EXPERIMENTS_1_TEST = [
    {"train": [1, 2, 3, 4], "test": [5], "name": "exp01_train1234_test5"},
    {"train": [1, 2, 3, 5], "test": [4], "name": "exp02_train1235_test4"},
    {"train": [1, 2, 4, 5], "test": [3], "name": "exp03_train1245_test3"},
    {"train": [1, 3, 4, 5], "test": [2], "name": "exp04_train1345_test2"},
    {"train": [2, 3, 4, 5], "test": [1], "name": "exp05_train2345_test1"},
]

# Experimentos com 2 amostras de teste
EXPERIMENTS_2_TEST = [
    {"train": [1, 2, 3], "test": [4, 5], "name": "exp06_train123_test45"},
    {"train": [1, 2, 4], "test": [3, 5], "name": "exp07_train124_test35"},
    {"train": [1, 3, 4], "test": [2, 5], "name": "exp08_train134_test25"},
    {"train": [2, 3, 4], "test": [1, 5], "name": "exp09_train234_test15"},
    {"train": [1, 2, 5], "test": [3, 4], "name": "exp10_train125_test34"},
    {"train": [1, 3, 5], "test": [2, 4], "name": "exp11_train135_test24"},
    {"train": [2, 3, 5], "test": [1, 4], "name": "exp12_train235_test14"},
    {"train": [1, 4, 5], "test": [2, 3], "name": "exp13_train145_test23"},
    {"train": [2, 4, 5], "test": [1, 3], "name": "exp14_train245_test13"},
    {"train": [3, 4, 5], "test": [1, 2], "name": "exp15_train345_test12"},
]

ALL_EXPERIMENTS = EXPERIMENTS_1_TEST + EXPERIMENTS_2_TEST


class ExperimentRunner:
    def __init__(self):
        self.base_saves_path = os.getenv('SAVES_PATH')
        self.results_dir = os.path.join(self.base_saves_path, 'experiments')
        os.makedirs(self.results_dir, exist_ok=True)

        self.all_results = []

    def run_all_experiments(self, skip_preprocessing=False):
        """
        Executa todos os experimentos.

        Args:
            skip_preprocessing: Se True, pula pré-processamento (se já foi feito)
        """
        print("="*80)
        print("EXECUTANDO EXPERIMENTOS COMPLETOS DE VALIDAÇÃO CRUZADA")
        print("="*80)
        print(f"Total de experimentos: {len(NUM_MFCC_OPTIONS)} MFCCs × {len(ALL_EXPERIMENTS)} divisões = {len(NUM_MFCC_OPTIONS) * len(ALL_EXPERIMENTS)}")
        print(f"Resultados serão salvos em: {self.results_dir}")
        print("="*80)

        experiment_count = 0
        total_experiments = len(NUM_MFCC_OPTIONS) * len(ALL_EXPERIMENTS)

        for num_mfccs in NUM_MFCC_OPTIONS:
            print(f"\n{'='*80}")
            print(f"PROCESSANDO COM {num_mfccs} MFCCs")
            print(f"{'='*80}")

            # Passo 1: Pré-processamento (se necessário)
            if not skip_preprocessing:
                self.run_preprocessing(num_mfccs)

            for exp_config in ALL_EXPERIMENTS:
                experiment_count += 1

                print(f"\n{'-'*80}")
                print(f"EXPERIMENTO {experiment_count}/{total_experiments}: {exp_config['name']}")
                print(f"MFCCs: {num_mfccs}")
                print(f"Treino: {exp_config['train']}")
                print(f"Teste: {exp_config['test']}")
                print(f"{'-'*80}")

                result = self.run_single_experiment(
                    num_mfccs=num_mfccs,
                    train_utterances=exp_config['train'],
                    test_utterances=exp_config['test'],
                    exp_name=exp_config['name']
                )

                self.all_results.append(result)

                # Salvar resultados parciais
                self.save_results_summary()

        # Gerar relatório final
        self.generate_final_report()

        print("\n" + "="*80)
        print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
        print(f"Resultados salvos em: {self.results_dir}")
        print("="*80)

    def run_preprocessing(self, num_mfccs):
        """Executa pré-processamento com número específico de MFCCs."""
        print(f"\nPré-processando com {num_mfccs} MFCCs...")

        # Atualizar variável de ambiente temporariamente
        os.environ['NUM_MFCCS'] = str(num_mfccs)

        # Criar subsistema e processar
        preprocessing_subsys = PreprocessingSubsystem()
        preprocessing_subsys.preprocess_signal()

        print(f"Pré-processamento com {num_mfccs} MFCCs concluído!")

    def run_single_experiment(self, num_mfccs, train_utterances, test_utterances, exp_name):
        """Executa um único experimento."""

        # Atualizar variável de ambiente
        os.environ['NUM_MFCCS'] = str(num_mfccs)

        # Criar subsistemas
        feature_adj_subsys = FeatureAdjustmentSubsystem()
        deep_learning_subsys = DeepLearningSubsystem()

        # Preparar dados com divisão específica
        print("\nPreparando dados...")
        training_data, test_data, training_labels, test_labels = self.prepare_data_custom_split(
            feature_adj_subsys,
            train_utterances,
            test_utterances,
            num_mfccs
        )

        print(f"Dados de treino: {training_data.shape}")
        print(f"Dados de teste: {test_data.shape}")

        # Treinar modelo
        print("\nTreinando modelo...")
        deep_learning_subsys.treinarModelo(
            training_data,
            training_labels,
            test_data,
            test_labels
        )

        # Avaliar modelo
        print("\nAvaliando modelo...")
        accuracy, predicted_labels = deep_learning_subsys.avaliarModelo(
            test_data,
            test_labels
        )

        # Criar diretório para este experimento
        exp_dir = os.path.join(self.results_dir, f"mfcc{num_mfccs}", exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Salvar modelo
        model_path = os.path.join(exp_dir, 'model.keras')
        deep_learning_subsys.model.save(model_path)

        # Mover gráficos para diretório do experimento
        if os.path.exists(os.path.join(self.base_saves_path, 'training_curves.png')):
            import shutil
            shutil.move(
                os.path.join(self.base_saves_path, 'training_curves.png'),
                os.path.join(exp_dir, 'training_curves.png')
            )
        if os.path.exists(os.path.join(self.base_saves_path, 'confusion_matrix.png')):
            import shutil
            shutil.move(
                os.path.join(self.base_saves_path, 'confusion_matrix.png'),
                os.path.join(exp_dir, 'confusion_matrix.png')
            )

        # Coletar métricas
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)

        # Salvar resultados
        result = {
            'experiment_name': exp_name,
            'num_mfccs': num_mfccs,
            'train_utterances': train_utterances,
            'test_utterances': test_utterances,
            'train_samples': len(training_data),
            'test_samples': len(test_data),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }

        # Salvar resultado individual
        result_file = os.path.join(exp_dir, 'result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"\nResultados salvos em: {exp_dir}")
        print(f"Acurácia: {accuracy*100:.2f}%")

        return result

    def prepare_data_custom_split(self, feature_adj_subsys, train_utterances, test_utterances, num_mfccs):
        """
        Prepara dados com divisão customizada de treino/teste.
        """
        # Encontrar número máximo de frames
        max_frames = 0
        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES')) + 1):
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
                frames = feature_adj_subsys.get_frames(mfccs_path, 'mfccs.npy')
                if frames > max_frames:
                    max_frames = frames

        print(f"Número máximo de frames: {max_frames}")

        # Equalizar frames
        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES')) + 1):
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
                feature_adj_subsys.equalize_frames(max_frames, mfccs_path, 'mfccs.npy')

        # Padronizar (concatenar)
        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES')) + 1):
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}')
                npy_list = [f'mfccs_{max_frames}.npy']
                feature_adj_subsys.standardize(mfccs_path, npy_list)

        # Organizar dados com divisão customizada
        training_data = []
        test_data = []
        training_labels = []
        test_labels = []

        for speaker in range(1, int(os.getenv('NUM_SPEAKERS')) + 1):
            for utterance in range(1, int(os.getenv('NUM_UTTERANCES')) + 1):
                mfccs_path = os.path.join(os.getenv('SAVES_PATH'), f'{speaker}', f'{utterance}/coefficients.npy')
                try:
                    normalized_mfccs = np.load(mfccs_path)

                    # Extrair apenas os MFCCs necessários
                    normalized_mfccs = normalized_mfccs[:num_mfccs, :]

                    if utterance in test_utterances:
                        test_data.append(normalized_mfccs)
                        test_labels.append(speaker - 1)
                    elif utterance in train_utterances:
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

        # Normalizar
        mean = np.mean(training_data, axis=(0, 2), keepdims=True)
        std = np.std(training_data, axis=(0, 2), keepdims=True)

        training_data = (training_data - mean) / std
        test_data = (test_data - mean) / std

        return training_data, test_data, training_labels, test_labels

    def save_results_summary(self):
        """Salva resumo de todos os resultados até o momento."""
        summary_file = os.path.join(self.results_dir, 'results_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(self.all_results, f, indent=4)

    def generate_final_report(self):
        """Gera relatório final com análise comparativa."""
        import pandas as pd

        # Criar DataFrame
        df = pd.DataFrame(self.all_results)

        # Relatório por número de MFCCs
        print("\n" + "="*80)
        print("RESUMO POR NÚMERO DE MFCCs")
        print("="*80)

        for num_mfccs in NUM_MFCC_OPTIONS:
            df_mfcc = df[df['num_mfccs'] == num_mfccs]
            print(f"\n{num_mfccs} MFCCs:")
            print(f"  Acurácia média: {df_mfcc['accuracy'].mean()*100:.2f}% ± {df_mfcc['accuracy'].std()*100:.2f}%")
            print(f"  F1-score médio: {df_mfcc['f1_score'].mean():.4f} ± {df_mfcc['f1_score'].std():.4f}")
            print(f"  Precision média: {df_mfcc['precision'].mean():.4f} ± {df_mfcc['precision'].std():.4f}")
            print(f"  Recall médio: {df_mfcc['recall'].mean():.4f} ± {df_mfcc['recall'].std():.4f}")

        # Relatório por tipo de experimento (1 vs 2 amostras de teste)
        print("\n" + "="*80)
        print("RESUMO POR TIPO DE EXPERIMENTO")
        print("="*80)

        df['num_test_samples'] = df['experiment_name'].apply(
            lambda x: 1 if int(x.split('_')[0].replace('exp', '')) <= 5 else 2
        )

        print("\n1 amostra de teste (leave-one-out):")
        df_1test = df[df['num_test_samples'] == 1]
        print(f"  Acurácia média: {df_1test['accuracy'].mean()*100:.2f}% ± {df_1test['accuracy'].std()*100:.2f}%")

        print("\n2 amostras de teste:")
        df_2test = df[df['num_test_samples'] == 2]
        print(f"  Acurácia média: {df_2test['accuracy'].mean()*100:.2f}% ± {df_2test['accuracy'].std()*100:.2f}%")

        # Salvar CSV
        csv_file = os.path.join(self.results_dir, 'results_summary.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nResultados salvos em CSV: {csv_file}")

        # Melhor experimento
        best_exp = df.loc[df['accuracy'].idxmax()]
        print("\n" + "="*80)
        print("MELHOR EXPERIMENTO")
        print("="*80)
        print(f"Nome: {best_exp['experiment_name']}")
        print(f"MFCCs: {best_exp['num_mfccs']}")
        print(f"Treino: {best_exp['train_utterances']}")
        print(f"Teste: {best_exp['test_utterances']}")
        print(f"Acurácia: {best_exp['accuracy']*100:.2f}%")
        print(f"F1-score: {best_exp['f1_score']:.4f}")


def main():
    runner = ExperimentRunner()

    # Perguntar se deve pular pré-processamento
    print("Deseja pular o pré-processamento?")
    print("(Digite 's' se os dados já foram pré-processados para todos os MFCCs)")
    skip = input("Pular pré-processamento? (s/n): ").lower() == 's'

    runner.run_all_experiments(skip_preprocessing=skip)


if __name__ == "__main__":
    main()
