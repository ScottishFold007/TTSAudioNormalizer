import os
import sox
import logging
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class AudioProcessingParams:
    """Audio Processing Parameters Configuration Class"""
    # Output format parameters
    output_format: str = 'wav'    # WAV format is lossless but larger, MP3 is lossy but smaller

    # Normalization parameters
    target_db: float = -3.0      # Target volume level (dB), 0dB is maximum, -3dB leaves headroom to prevent clipping

    # Noise reduction parameters
    noise_reduction_enabled: bool = True  # Enable noise reduction processing
    noise_threshold_db: float = -40.0    # Sound below this dB value is treated as noise
    noise_attack_time: float = 0.02      # Time (seconds) for noise reduction to take effect
    noise_release_time: float = 0.1      # Fade-out time (seconds) for noise reduction

    # Silence detection parameters
    silence_threshold: float = 2.0       # Volume threshold for silence detection (percentage of max volume)
    min_silence_duration: float = 0.1    # Minimum silence duration (seconds) to be detected

    # Compressor parameters
    compand_enabled: bool = True         # Enable dynamic range compression
    attack_time: float = 0.1            # Compressor attack time (seconds)
    decay_time: float = 0.2             # Compressor release time (seconds)
    soft_knee_db: float = 2.0           # Compression transition range (dB)
    compression_ratio: float = 3.0       # Compression ratio, e.g., 2.5:1 means 2.5dB input increase yields 1dB output increase

    # Equalizer parameters
    equalizer_enabled: bool = True       # Enable equalizer adjustment

    # High frequency settings (2kHz - 8kHz)
    treble_gain: float = 2.0            # High frequency gain (dB)
    treble_slope: float = 0.5           # High frequency slope
    treble_frequency: float = 3000.0    # High frequency center frequency (Hz)

    # Mid frequency settings (250Hz - 2kHz)
    mid_gain: float = 1.0               # Mid frequency gain (dB)
    mid_frequency: float = 1000.0       # Mid frequency center frequency (Hz)
    mid_q: float = 0.707                # Q value, affects mid frequency bandwidth

    # Low frequency settings (20Hz - 250Hz)
    bass_gain: float = 3.0              # Low frequency gain (dB)
    bass_slope: float = 0.4             # Low frequency slope
    bass_frequency: float = 100.0       # Low frequency center frequency (Hz)

    # Subsonic control (below 20Hz)
    subsonic_filter_enabled: bool = True # Enable subsonic filtering
    subsonic_frequency: float = 20.0     # Subsonic cutoff frequency (Hz)

    # Fade effects
    fade_enabled: bool = True           # Enable fade in/out effects
    fade_in_time: float = 0.02          # Fade in time (seconds)
    fade_out_time: float = 0.02         # Fade out time (seconds)

    # Sample rate and channels
    rate_enabled: bool = True           
    rate: int = 22050                   # Sample rate (Hz), 22050Hz suitable for speech
    channels_enabled: bool = True        
    channels: int = 1                   # Number of channels, 1 for mono (suitable for speech), 2 for stereo


class AudioNormalizer:
    """Base Audio Normalization Class"""
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
        self.failed_files: List[Tuple[str, str]] = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'audio_normalize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
    
    def _validate_input_file(self, input_path: str) -> None:
        """Validate input file"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        
        if os.path.getsize(input_path) == 0:
            raise ValueError(f"File size is 0: {input_path}")
            
        if not self.check_audio_format(input_path):
            raise ValueError(f"Unsupported audio format: {input_path}")

    def _get_audio_files(self, input_dir: str, output_dir: str, params: AudioProcessingParams) -> List[Tuple[str, str]]:
        """Obtain the list of audio files that need to be processed"""
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if self.check_audio_format(file):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, input_dir)
                    base_name = os.path.splitext(rel_path)[0]
                    output_path = os.path.join(output_dir, f"{base_name}.{params.output_format}")
                    audio_files.append((input_path, output_path))
        return audio_files

    def _handle_error(self, input_path: str, error: Exception) -> None:
        """Handle errors"""
         error_msg = f"Failed to process file {input_path}: {str(error)}"
        logging.error(error_msg)
        self.failed_files.append((input_path, str(error)))

    def check_audio_format(self, file_path: str) -> bool:
        """Check if the audio format is supported."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_formats

    def _print_summary(self, total_files: int) -> None:
        """Print processing results statistics."""
        failed_count = len(self.failed_files)
        success_count = total_files - failed_count
        
        print("\nProcessing completion statistics:")
        print(f"Total number of files: {total_files}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {failed_count}")
        
        if failed_count > 0:
            print("\nList of failed files:")
            for file_path, error in self.failed_files:
                print(f"- {file_path}: {error}")


class TTSAudioNormalizer(AudioNormalizer):
    """A dedicated audio standardization processor for TTS training."""
    
    def __init__(self):
        super().__init__()
        self.quality_metrics: Dict[str, Any] = {}
    
 
    def normalize_audio(self, 
                       input_path: str, 
                       output_path: str, 
                       params: Optional[AudioProcessingParams] = None) -> bool:
        """
        Optimized audio processing workflow specifically for TTS training.
        """
        if params is None:
            params = AudioProcessingParams()

        try:
            self._validate_input_file(input_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            transformer = sox.Transformer()

            # 1. Basic Preprocessing
            if params.rate_enabled:
                transformer.rate(params.rate)
            if params.channels_enabled:
                transformer.channels(params.channels)
            transformer.dcshift(shift=0.0)  # Eliminate DC offset.

            # 2. Noise and silence processing.
            if params.noise_reduction_enabled:
                # Multi-stage compression processing.
                transformer.compand(
                    attack_time=params.noise_attack_time,
                decay_time=params.noise_release_time,
                    soft_knee_db=2.0,
                    db_level=[-60, -30, -20],
                    post_gain_db=0
                )

                # Noise gate processing.
                transformer.noisegate(
                    threshold_db=params.noise_threshold_db,
                    attack_time=params.noise_attack_time,
                    release_time=params.noise_release_time
                )

        
            # Silence processing
            # Remove silence at the beginning
            

            transformer.silence(
                location=1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )

            # Remove silence at the end
            transformer.silence(
                location=-1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )
        
            # Remove silence segments in the middle
            transformer.silence(
                location=0,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )

            # 3. Frequency response optimization
            if params.subsonic_filter_enabled:
                transformer.highpass(params.subsonic_frequency, 2.0)

            if params.equalizer_enabled:
                if params.bass_gain != 0:
                    transformer.bass(
                        gain_db=params.bass_gain,
                        frequency=params.bass_frequency,
                        slope=params.bass_slope
                    )

                transformer.equalizer(
                    frequency=params.mid_frequency,
                    width_q=params.mid_q,
                    gain_db=params.mid_gain
                )

                if params.treble_gain != 0:
                    transformer.treble(
                        gain_db=params.treble_gain,
                        frequency=params.treble_frequency,
                        slope=params.treble_slope
                    )

            # 4. Dynamic range processing.
            if params.compand_enabled:
                # Perform dynamic range compression first
                transformer.compand(
                    attack_time=params.attack_time,
                    decay_time=params.decay_time,
                    soft_knee_db=params.soft_knee_db,
                    threshold_db=-20,
                    compression_ratio=params.compression_ratio
                )

            # Normalize the volume at the end
            transformer.norm(params.target_db)

            # 5. Edge processing
            if params.fade_enabled:
                transformer.fade(
                    fade_in_len=params.fade_in_time,
                    fade_out_len=params.fade_out_time
                )

            # 6. Perform the transformation
            transformer.build(input_path, output_path)

            # 7. Quality inspection
            self._check_audio_quality(output_path)
        
            logging.info(f"Successfully processed files: {input_path}")
            return True

        except Exception as e:
            self._handle_error(input_path, e)
            return False



            
    def _check_audio_quality(self, audio_path: str) -> None:
        """
       Inspect the quality of the processed audio
        """
        try:
            # Loading audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate quality metrics
            metrics = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossings': int(librosa.zero_crossings(y).sum()),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'silence_ratio': float(np.mean(np.abs(y) < 0.01))
            }
            
            # Store quality metrics
            self.quality_metrics[audio_path] = metrics
            
            # Check if it meets the requirements
            if metrics['duration'] < 0.1:  # Too short audio
                raise ValueError("The audio is too short")
            if metrics['rms_energy'] < 0.01:  # Too quiet
                raise ValueError("The volume is too low")
            if metrics['silence_ratio'] > 0.5:  # Too much silence
                raise ValueError("The silence ratio is too high")

                
        except Exception as e:
            logging.warning(f"Quality check failed {audio_path}: {str(e)}")
            
    def get_quality_report(self) -> Dict[str, Dict[str, float]]:
        """
        Get audio quality report
        """
        return self.quality_metrics
        
    def batch_normalize_directory(self, 
                                input_dir: str, 
                                output_dir: str, 
                                params: Optional[AudioProcessingParams] = None,
                                max_workers: int = 4) -> None:
        """
        Batch process audio files in directory
        """
        if params is None:
            params = AudioProcessingParams()
    
        if params.output_format.lower() not in [fmt.strip('.') for fmt in self.supported_formats]:
            raise ValueError(f"Unsupported output format: {params.output_format}")
    
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
        audio_files = self._get_audio_files(input_dir, output_dir, params)
        total_files = len(audio_files)
    
        # Use a progress bar object
        pbar = tqdm(total=total_files, desc="Processing Progress", ncols=100, position=1, leave=False)
    
        def process_file(input_path, output_path):
            result = self.normalize_audio(input_path, output_path, params)
            pbar.update(1)
            return result
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_path, output_path in audio_files:
                future = executor.submit(process_file, input_path, output_path)
                futures.append(future)
    
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
        pbar.close()
        self._print_summary(total_files)
        self._print_quality_summary()
        
    def _print_quality_summary(self) -> None:
        """
        Print audio quality statistics report
        """
        if not self.quality_metrics:
            return
            
        print("\nAudio Quality Statistics:")
        metrics_summary = {
            'duration': [],
            'rms_energy': [],
            'silence_ratio': []
        }
        
        for metrics in self.quality_metrics.values():
            metrics_summary['duration'].append(metrics['duration'])
            metrics_summary['rms_energy'].append(metrics['rms_energy'])
            metrics_summary['silence_ratio'].append(metrics['silence_ratio'])
            
        for metric_name, values in metrics_summary.items():
            values = np.array(values)
            print(f"\n{metric_name} statistics:")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Std Dev: {np.std(values):.3f}")
            print(f"  Min: {np.min(values):.3f}")
            print(f"  Max: {np.max(values):.3f}")
    
