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
    """音频处理参数配置类"""
    # 输出格式参数
    output_format: str = 'wav'    # wav格式无损但文件较大，mp3有损但文件小

    # 标准化参数
    target_db: float = -3.0      # 目标音量大小(dB)，0dB是最大值，-3dB留有余量防止削峰，值越小音量越小

    # 降噪参数
    noise_reduction_enabled: bool = True  # 是否启用降噪处理
    noise_threshold_db: float = -40.0    # 低于此分贝值的声音视为噪声被处理，值越大降噪越强但可能影响有效声音；提高噪声阈值
    noise_attack_time: float = 0.02      # 降噪开始作用的时间(秒)，值越小响应越快但可能产生不自然感；降低起始时间使降噪更快响应
    noise_release_time: float = 0.1     # 降噪结束的渐变时间(秒)，值越大过渡越自然但可能使噪声残留时间变长；增加释放时间使声音更自然

    # 静音检测参数
    silence_threshold: float = 2.0       # 判定为静音的音量阈值，占最大音量的百分比，值越大越容易判定为静音
    min_silence_duration: float = 0.1    # 最短静音判定时长(秒)，小于此时长的静音将被保留

    # 压限器参数
    compand_enabled: bool = True         # 是否启用动态范围压缩
    attack_time: float = 0.1            # 压缩器启动时间(秒)，影响对突发声音的响应速度
    decay_time: float = 0.2               # 压缩器释放时间(秒)，影响声音衰减的自然度
    soft_knee_db: float = 2.0            # 压缩过渡区间(dB)，值越大压缩更平滑但不够干脆
    compression_ratio: float = 3.0        # 压缩比率，如2.5:1表示输入增加2.5dB时输出只增加1dB

    # 均衡器参数
    equalizer_enabled: bool = True       # 是否启用均衡器调节

    # 高频段设置 (2kHz - 8kHz)
    treble_gain: float = 2.0            # 高频增益(dB)，正值增强清晰度，负值减弱刺耳感
    treble_slope: float = 0.5           # 高频斜率，值越大频率响应曲线越陡峭
    treble_frequency: float = 3000.0    # 高频中心频率(Hz)，调节高频开始作用的位置

    # 中频段设置 (250Hz - 2kHz)
    mid_gain: float = 1.0               # 中频增益(dB)，影响人声主要频段的突出程度
    mid_frequency: float = 1000.0       # 中频中心频率(Hz)，人声最敏感区域约在1kHz
    mid_q: float = 0.707                # Q值，影响中频段的带宽，值越大带宽越窄

    # 低频段设置 (20Hz - 250Hz)
    bass_gain: float = 3.0              # 低频增益(dB)，影响声音厚重感
    bass_slope: float = 0.4             # 低频斜率，值越大频率响应曲线越陡峭
    bass_frequency: float = 100.0       # 低频中心频率(Hz)，调节低频开始作用的位置

    # 亚音频控制 (20Hz以下)
    subsonic_filter_enabled: bool = True # 是否启用次低音过滤
    subsonic_frequency: float = 20.0     # 次低音截止频率(Hz)，低于此频率的声音将被过滤

    # 淡入淡出
    fade_enabled: bool = True           # 是否启用淡入淡出效果
    fade_in_time: float = 0.02          # 淡入时间(秒)，防止开始的爆音
    fade_out_time: float = 0.02         # 淡出时间(秒)，防止结束的爆音

    # 采样率和声道
    rate_enabled: bool = True           
    rate: int = 22050                   # 采样率(Hz)，22050Hz适合语音，更高的值增加文件大小
    channels_enabled: bool = True        
    channels: int = 1                   # 声道数，1为单声道(适合语音)，2为立体声


class AudioNormalizer:
    """基础音频标准化类"""
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
        self.failed_files: List[Tuple[str, str]] = []
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'audio_normalize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
    
    def _validate_input_file(self, input_path: str) -> None:
        """验证输入文件"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        if os.path.getsize(input_path) == 0:
            raise ValueError(f"文件大小为0: {input_path}")
            
        if not self.check_audio_format(input_path):
            raise ValueError(f"不支持的音频格式: {input_path}")

    def _get_audio_files(self, input_dir: str, output_dir: str, params: AudioProcessingParams) -> List[Tuple[str, str]]:
        """获取需要处理的音频文件列表"""
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
        """处理错误"""
        error_msg = f"处理文件 {input_path} 失败: {str(error)}"
        logging.error(error_msg)
        self.failed_files.append((input_path, str(error)))

    def check_audio_format(self, file_path: str) -> bool:
        """检查音频格式是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_formats

    def _print_summary(self, total_files: int) -> None:
        """打印处理结果统计"""
        failed_count = len(self.failed_files)
        success_count = total_files - failed_count
        
        print("\n处理完成统计:")
        print(f"总文件数: {total_files}")
        print(f"成功处理: {success_count}")
        print(f"处理失败: {failed_count}")
        
        if failed_count > 0:
            print("\n失败文件列表:")
            for file_path, error in self.failed_files:
                print(f"- {file_path}: {error}")


class TTSAudioNormalizer(AudioNormalizer):
    """专门针对TTS训练的音频标准化处理器"""
    
    def __init__(self):
        super().__init__()
        self.quality_metrics: Dict[str, Any] = {}
    
 
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

            transformer = sox.Transformer()

            # 1. 基础预处理
            if params.rate_enabled:
                transformer.rate(params.rate)
            if params.channels_enabled:
                transformer.channels(params.channels)
            transformer.dcshift(shift=0.0)  # 消除直流偏移

            # 2. 降噪和静音处理
            if params.noise_reduction_enabled:
                # 多级压缩处理
                transformer.compand(
                    attack_time=params.noise_attack_time,
                decay_time=params.noise_release_time,
                    soft_knee_db=2.0,
                    db_level=[-60, -30, -20],
                    post_gain_db=0
                )

                # 噪声门处理
                transformer.noisegate(
                    threshold_db=params.noise_threshold_db,
                    attack_time=params.noise_attack_time,
                    release_time=params.noise_release_time
                )

            # 静音处理
            # 去除开头的静音
            transformer.silence(
                location=1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )

            # 去除结尾的静音
            transformer.silence(
                location=-1,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )
        
            # 去除中间的静音段
            transformer.silence(
                location=0,
                silence_threshold=params.silence_threshold,
                min_silence_duration=params.min_silence_duration,
                buffer_around_silence=True
            )

            # 3. 频率响应优化
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

            # 4. 动态范围处理
            if params.compand_enabled:
                # 先进行动态范围压缩
                transformer.compand(
                    attack_time=params.attack_time,
                    decay_time=params.decay_time,
                    soft_knee_db=params.soft_knee_db,
                    threshold_db=-20,
                    compression_ratio=params.compression_ratio
                )

            # 最后进行音量标准化
            transformer.norm(params.target_db)

            # 5. 边缘处理
            if params.fade_enabled:
                transformer.fade(
                    fade_in_len=params.fade_in_time,
                    fade_out_len=params.fade_out_time
                )

            # 6. 执行转换
            transformer.build(input_path, output_path)

            # 7. 质量检查
            self._check_audio_quality(output_path)
        
            logging.info(f"成功处理文件: {input_path}")
            return True

        except Exception as e:
            self._handle_error(input_path, e)
            return False



            
    def _check_audio_quality(self, audio_path: str) -> None:
        """
        检查处理后的音频质量
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None)
            
            # 计算质量指标
            metrics = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossings': int(librosa.zero_crossings(y).sum()),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'silence_ratio': float(np.mean(np.abs(y) < 0.01))
            }
            
            # 存储质量指标
            self.quality_metrics[audio_path] = metrics
            
            # 检查是否符合要求
            if metrics['duration'] < 0.1:  # 太短的音频
                raise ValueError("音频太短")
            if metrics['rms_energy'] < 0.01:  # 音量太小
                raise ValueError("音量太小")
            if metrics['silence_ratio'] > 0.5:  # 静音太多
                raise ValueError("静音比例过高")
                
        except Exception as e:
            logging.warning(f"质量检查失败 {audio_path}: {str(e)}")
            
    def get_quality_report(self) -> Dict[str, Dict[str, float]]:
        """
        获取音频质量报告
        """
        return self.quality_metrics
        
    def batch_normalize_directory(self, 
                                input_dir: str, 
                                output_dir: str, 
                                params: Optional[AudioProcessingParams] = None,
                                max_workers: int = 4) -> None:
        """
        批量处理目录中的音频文件
        """
        if params is None:
            params = AudioProcessingParams()

        if params.output_format.lower() not in [fmt.strip('.') for fmt in self.supported_formats]:
            raise ValueError(f"不支持的输出格式: {params.output_format}")

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        audio_files = self._get_audio_files(input_dir, output_dir, params)
        total_files = len(audio_files)

        # 使用一个进度条对象
        pbar = tqdm(total=total_files, desc="处理进度", ncols=100, position=1, leave=False)

        def process_file(input_path, output_path):
            result = self.normalize_audio(input_path, output_path, params)
            pbar.update(1)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_path, output_path in audio_files:
                future = executor.submit(process_file, input_path, output_path)
                futures.append(future)

            # 等待所有任务完成
            for future in futures:
                future.result()

        pbar.close()
        self._print_summary(total_files)
        self._print_quality_summary()
        
    def _print_quality_summary(self) -> None:
        """
        打印音频质量统计报告
        """
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

