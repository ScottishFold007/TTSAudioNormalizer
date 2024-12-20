import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf
import sox
import noisereduce as nr
from tqdm import tqdm

@dataclass
class AudioProcessingParams:
    def __init__(self):
        # 基础参数
        self.rate = 44100
        self.channels = 1
        self.rate_enabled = True
        self.channels_enabled = True
        
        # 降噪参数
        self.noise_reduction_enabled = True
        self.noise_reduction_strength = 0.7
        self.noise_attack_time = 0.02
        self.noise_release_time = 0.02
        
        # 静音处理参数
        self.silence_threshold = 0.1
        self.min_silence_duration = 0.3
        
        # 频率响应参数
        self.subsonic_filter_enabled = True
        self.subsonic_frequency = 60
        self.equalizer_enabled = True
        self.bass_gain = 0
        self.bass_frequency = 100
        self.bass_slope = 0.5
        self.mid_frequency = 1000
        self.mid_q = 0.707
        self.mid_gain = 0
        self.treble_gain = 0
        self.treble_frequency = 5000
        self.treble_slope = 0.5
        
        # 动态范围参数
        self.compand_enabled = True
        self.attack_time = 0.02
        self.decay_time = 0.15
        self.soft_knee_db = 6.0
        self.target_db = -3
        
        # 淡入淡出参数
        self.fade_enabled = True
        self.fade_in_time = 0.01
        self.fade_out_time = 0.01
        
        # 输出格式
        self.output_format = "wav"     


class TTSAudioNormalizer:
    """专门针对TTS训练的音频标准化处理器"""
    
    def __init__(self):
        self.quality_metrics: Dict[str, Any] = {}
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg']
        self.processed_count = 0
        self.error_count = 0
    
    def normalize_audio(self, 
                       input_path: str, 
                       output_path: str, 
                       params: Optional[AudioProcessingParams] = None) -> bool:
        """
        优化的音频处理流程，专门针对TTS训练
        """
        if params is None:
            params = AudioProcessingParams()

        try:
            self._validate_input_file(input_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 1. 首先进行降噪处理
            temp_path = None
            if params.noise_reduction_enabled:
                # 读取音频文件
                y, sr = librosa.load(input_path, sr=None)
                
                # 使用noisereduce进行降噪
                reduced_noise = nr.reduce_noise(
                    y=y,
                    sr=sr,
                    prop_decrease=params.noise_reduction_strength,
                    stationary=True,
                    n_jobs=-1
                )
                
                # 保存临时文件
                temp_path = output_path + '.temp.wav'
                sf.write(temp_path, reduced_noise, sr)
                
                # 更新input_path为降噪后的临时文件
                processing_input = temp_path
            else:
                processing_input = input_path

            # 2. 使用sox进行其他处理
            transformer = sox.Transformer()

            # 基础预处理
            if params.rate_enabled:
                transformer.rate(params.rate)
            if params.channels_enabled:
                transformer.channels(params.channels)
            transformer.dcshift(shift=0.0)

            # 动态范围压缩
            if params.compand_enabled:
                transformer.compand(
                    attack_time=params.attack_time,
                    decay_time=params.decay_time,
                    soft_knee_db=params.soft_knee_db
                )

            # 静音处理
            transformer.silence(  # 去除开头的静音
                location=1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )
            transformer.silence(  # 去除结尾的静音
                location=-1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )
            transformer.silence(  # 去除中间的静音段
                location=0,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )

            # 频率响应优化
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

            # 音量标准化
            transformer.norm(params.target_db)

            # 淡入淡出
            if params.fade_enabled:
                transformer.fade(
                    fade_in_len=params.fade_in_time,
                    fade_out_len=params.fade_out_time
                )

            # 执行转换
            transformer.build(processing_input, output_path)

            # 删除临时文件
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            # 质量检查
            self._check_audio_quality(output_path)
            
            self.processed_count += 1
            logging.info(f"成功处理文件: {input_path}")
            return True

        except Exception as e:
            self._handle_error(input_path, e)
            self.error_count += 1
            return False

    def _validate_input_file(self, input_path: str) -> None:
        """验证输入文件"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        if not any(input_path.lower().endswith(fmt) for fmt in self.supported_formats):
            raise ValueError(f"不支持的文件格式: {input_path}")

    def _handle_error(self, input_path: str, error: Exception) -> None:
        """处理错误"""
        logging.error(f"处理文件失败 {input_path}: {str(error)}")

    def _check_audio_quality(self, audio_path: str) -> None:
        """检查处理后的音频质量"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            metrics = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossings': int(librosa.zero_crossings(y).sum()),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'silence_ratio': float(np.mean(np.abs(y) < 0.01))
            }
            
            self.quality_metrics[audio_path] = metrics
            
            if metrics['duration'] < 0.1:
                raise ValueError("音频太短")
            if metrics['rms_energy'] < 0.01:
                raise ValueError("音量太小")
            if metrics['silence_ratio'] > 0.5:
                raise ValueError("静音比例过高")
                
        except Exception as e:
            logging.warning(f"质量检查失败 {audio_path}: {str(e)}")

    def _get_audio_files(self, input_dir: str, output_dir: str, params: AudioProcessingParams) -> List[Tuple[str, str]]:
        """获取需要处理的音频文件列表"""
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in self.supported_formats):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(
                        output_dir,
                        os.path.splitext(rel_path)[0] + '.' + params.output_format
                    )
                    audio_files.append((input_path, output_path))
        return audio_files

    def batch_normalize_directory(self, 
                                input_dir: str, 
                                output_dir: str, 
                                params: Optional[AudioProcessingParams] = None,
                                max_workers: int = 4) -> None:
        """批量处理目录中的音频文件"""
        if params is None:
            params = AudioProcessingParams()

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        audio_files = self._get_audio_files(input_dir, output_dir, params)
        total_files = len(audio_files)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=total_files, desc="处理进度", ncols=100) as pbar:
                def process_file(args):
                    input_path, output_path = args
                    result = self.normalize_audio(input_path, output_path, params)
                    pbar.update(1)
                    return result

                list(executor.map(process_file, audio_files))

        self._print_summary(total_files)
        self._print_quality_summary()

    def _print_summary(self, total_files: int) -> None:
        """打印处理总结"""
        print("\n处理完成:")
        print(f"总文件数: {total_files}")
        print(f"成功处理: {self.processed_count}")
        print(f"处理失败: {self.error_count}")

    def _print_quality_summary(self) -> None:
        """打印音频质量统计报告"""
        if not self.quality_metrics:
            return
            
        print("\n音频质量统计:")
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
            print(f"\n{metric_name}统计:")
            print(f"  平均值: {np.mean(values):.3f}")
            print(f"  标准差: {np.std(values):.3f}")
            print(f"  最小值: {np.min(values):.3f}")
            print(f"  最大值: {np.max(values):.3f}")
