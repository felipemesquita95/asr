from dotenv import load_dotenv
import librosa
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, resample
import gc

load_dotenv()

class PreprocessingSubsystem:
    def __init__(self):
        self.audio_path = os.getenv('UTTERANCES_PATH')
        self.output_base_path = os.getenv('SAVES_PATH')
        self.sampling_rate = int(os.getenv('UTTERANCES_SAMPLING_RATE'))
        self.num_speakers = int(os.getenv('NUM_SPEAKERS'))
        self.num_utterances = int(os.getenv('NUM_UTTERANCES'))
        self.num_mfccs = int(os.getenv('NUM_MFCCS'))
        self.frame_size = int(os.getenv('FRAME_SIZE'))

        print("Inicializando subsistema de pré-processamento...")


    def pre_emphasis(self, audio_signal, coef: float = 0.97):
        return np.append(audio_signal[0], audio_signal[1:] - coef * audio_signal[:-1])

    def load_audio(self, filepath):
        audio, sr = librosa.load(filepath, sr=self.sampling_rate)
        return audio, sr

    def plot_time_domain(self, audio, sr, output_path):
        time = np.arange(0, len(audio)) / sr
        plt.figure(figsize=(10, 6))
        plt.plot(time, audio, color='blue')
        plt.title('Arquivo .WAV no domínio do tempo', fontsize=12)
        plt.xlabel('Tempo(s)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True)
        plt.xlim(0, len(audio) / sr)
        plt.savefig(output_path, dpi=300)
        plt.close()

    def plot_frequency_spectrum(self, audio, sr, output_path, title):
        spectrum = np.abs(np.fft.fft(audio))
        frequency = np.fft.fftfreq(len(spectrum), d=1 / sr)
        mask = frequency >= 0
        frequency = frequency[mask]
        spectrum = spectrum[mask]

        plt.figure(figsize=(10, 6))
        plt.plot(frequency / 1000, spectrum, color='blue')
        plt.title(title, fontsize=12)
        plt.xlabel('Frequência (kHz)', fontsize=10)
        plt.ylabel('Magnitude', fontsize=10)
        plt.grid(True)
        plt.xlim(0, max(frequency / 1000))
        plt.savefig(output_path, dpi=300)
        plt.close()

    def low_pass_filter(self, audio, sr, cutoff=4000):
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, audio)

    def resample_audio(self, audio, orig_sr, target_sr):
        return resample(audio, int(len(audio) * target_sr / orig_sr))

    def extract_mfccs(self, audio, sr, num_mfccs, frame_size):
        hop_length = frame_size // 2
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_length, n_mfcc=num_mfccs)
        return mfccs

    def plot_mfccs(self, mfccs, output_path, title):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

    def preprocess_signal(self):
        for speaker in range(1, self.num_speakers + 1):
            for utterance in range(1, self.num_utterances + 1):
                
                utterance_index = (speaker - 1) * self.num_utterances + utterance
                utterance_path = os.path.join(self.audio_path, f'{utterance_index}.wav')
                
                if os.path.exists(utterance_path):
                    
                    output_path = os.path.join(self.output_base_path, f'{speaker}', f'{utterance}')
                    
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    audio, sr = self.load_audio(utterance_path)

                    self.plot_time_domain(audio, sr, os.path.join(output_path, 'dominioTempo.png'))
                    self.plot_frequency_spectrum(audio, sr, os.path.join(output_path, 'espectro48kHz.png'), 'Espectro de frequência do arquivo .WAV')

                    audio_filtered = self.low_pass_filter(audio, sr)
                    self.plot_frequency_spectrum(audio_filtered, sr, os.path.join(output_path, 'espectroFiltrado.png'), 'Espectro de frequência após filtragem antialiasing')

                    audio_pre_emphasized = self.pre_emphasis(audio_filtered)
                    self.plot_frequency_spectrum(audio_pre_emphasized, sr, os.path.join(output_path, 'espectroPE.png'), 'Espectro de frequência após pré-ênfase')

                    audio_resampled = self.resample_audio(audio_pre_emphasized, sr, 8000)
                    self.plot_frequency_spectrum(audio_resampled, 8000, os.path.join(output_path, 'espectro8kHz.png'), 'Espectro de frequência após reamostragem')

                    mfccs = self.extract_mfccs(audio_resampled, 8000, self.num_mfccs, self.frame_size)
                    self.plot_mfccs(mfccs, os.path.join(output_path, 'mfccs.png'), 'Matriz de MFCCs')

                    delta = librosa.feature.delta(mfccs)
                    self.plot_mfccs(delta, os.path.join(output_path, 'delta.png'), 'Matriz de deltas')

                    delta_delta = librosa.feature.delta(mfccs, order=2)
                    self.plot_mfccs(delta_delta, os.path.join(output_path, 'deltaDelta.png'), 'Matriz de deltas-deltas')

                    np.save(os.path.join(output_path, 'mfccs.npy'), mfccs)
                    np.save(os.path.join(output_path, 'delta.npy'), delta)
                    np.save(os.path.join(output_path, 'deltaDelta.npy'), delta_delta)

                    print(f'Resultados salvos para o locutor {speaker}, amostra {utterance}.')

                    audio = None
                    audio_filtered = None
                    audio_resampled = None
                    gc.collect()


