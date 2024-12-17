# TTSAudioNormalizer
TTSAudioNormalizer是一个专门用于TTS数据制作的工具，包括音频数据响度相关的描述性统计分析，响度归一化等操作。


# 音频标准化

TTS训练前进行音频标准化的意义主要有以下几个方面：

1. **提高训练稳定性**
```python
# 音量标准化的意义
target_db: float = -3.0      # 统一音量级别
                            # 避免模型因音量差异过大而学习到错误特征
                            # 减少梯度爆炸/消失的风险
```

2. **改善数据质量**
```python
# 频率响应优化的意义
equalizer_enabled: bool = True    # 突出语音关键频段
treble_frequency: float = 3000.0  # 增强辅音清晰度(2-8kHz)
mid_frequency: float = 1000.0     # 保持元音自然度(250Hz-2kHz)
bass_frequency: float = 100.0     # 控制基频范围(80-250Hz)
```

3. **减少噪声干扰**
```python
# 噪声控制的意义
subsonic_filter_enabled: bool = True  # 去除无用的超低频
compression_ratio: float = 2.5        # 压缩动态范围，降低背景噪声影响
threshold_db: float = -15.0           # 设置合理阈值，区分信号和噪声
```

4. **提升模型泛化能力**
```python
# 统一化处理的意义
rate: int = 44100            # 统一采样率
channels: int = 1            # 统一为单声道
output_format: str = 'wav'   # 统一音频格式
```

5. **加速训练收敛**
- 统一的数据格式和特征分布有助于模型更快收敛
- 减少模型学习数据预处理的负担
- 使模型更专注于学习语音合成的核心特征

6. **提高合成质量**
```python
# 音质优化的意义
attack_time: float = 0.15    # 保持音素边界清晰
decay_time: float = 0.8      # 维持自然的语音衰减
soft_knee_db: float = 3.0    # 确保音色过渡平滑
```

7. **数据集一致性**
```python
# 处理流程标准化
def normalize_dataset(self):
    """
    1. 统一格式和技术规格
    2. 统一响度级别
    3. 统一频率响应特征
    4. 统一动态范围
    5. 去除异常样本
    """
```

8. **降低后期处理成本**
- 预先处理可以减少训练后的音频后处理工作
- 提高模型直接输出的音频质量
- 简化部署流程

9. **特征提取优化**
```python
# 有利于特征提取的处理
mid_q: float = 0.707         # 确保频带划分合理
treble_slope: float = 0.5    # 保持谐波结构清晰
bass_slope: float = 0.4      # 避免基频提取困难
```

10. **实际应用考虑**
```python
# 应用场景适配
fade_enabled: bool = True     # 避免音频切分产生爆音
fade_in_time: float = 0.02   # 保持音素切分准确性
fade_out_time: float = 0.02  # 确保句子边界平滑
```

建议的标准化流程：

```python
class TTSAudioNormalizer:
    def process_pipeline(self, audio_file):
        """
        TTS训练前的音频标准化流程
        """
        # 1. 基础预处理
        self.convert_format()      # 统一格式
        self.resample()           # 统一采样率
        self.convert_channels()   # 转换为单声道
        
        # 2. 音质优化
        self.remove_dc_offset()   # 去除直流偏置
        self.normalize_volume()   # 音量标准化
        self.optimize_frequency() # 频率响应优化
        
        # 3. 噪声处理
        self.remove_silence()     # 去除静音段
        self.reduce_noise()       # 降噪处理
        self.compress_dynamic()   # 压缩动态范围
        
        # 4. 质量检查
        self.check_quality()      # 质量验证
        self.validate_features()  # 特征验证
```

注意事项：

1. **保持语音特征**
- 不要过度处理导致丢失关键语音特征
- 保持音素边界的清晰度
- 维持语音的自然韵律

2. **数据集特性**
- 根据数据集的特点调整参数
- 考虑说话人的声音特点
- 适应不同录音环境的需求

3. **模型需求**
- 根据具体TTS模型的需求调整处理方式
- 考虑特征提取算法的要求
- 适配训练框架的数据格式要求

通过合理的音频标准化，可以显著提高TTS训练的效率和效果。


##  核心参数设置策略

### 1. Mean Norm（平均归一化音量）
- **实际意义**：
  - 反映音频的整体响度水平
  - 表示音频信号的平均绝对振幅
  - 值域范围通常在0-1之间
- **数值含义**：
  - 值越大 = 整体听感越响
  - 值越小 = 整体听感越轻
  - 理想范围通常在0.1-0.3之间
- **应用场景**：
  - 用于评估音频的整体响度是否合适
  - 帮助判断是否需要音量增益

### 2. RMS Amplitude（均方根振幅）
- **实际意义**：
  - 反映音频的有效能量水平
  - 更接近人耳对响度的感知
  - 考虑了时间维度上的能量分布
- **数值含义**：
  - 值越大 = 音频能量越强
  - 值越小 = 音频能量越弱
  - 专业音频通常建议在0.1-0.4之间
- **应用场景**：
  - 评估音频的动态范围
  - 判断音频是否需要压缩或扩展
  - 常用于音频标准化处理

### 3. Max Amplitude（最大振幅）
- **实际意义**：
  - 反映音频中的峰值水平
  - 表示信号的最大瞬时值
  - 用于判断是否存在削波
- **数值含义**：
  - 1.0 = 达到数字音频的最大可能值（可能出现削波）
  - 建议峰值控制在0.9以下
  - 过低（如<0.5）表示音频可能过轻
- **应用场景**：
  - 检测音频是否存在失真
  - 评估音频的动态余量
  - 指导限幅器的设置

### 三者的关系
1. **层次关系**：
   - Max Amplitude > RMS Amplitude > Mean Norm
   - 这是由于它们计算方法的不同

2. **实际应用**：
   - Mean Norm：用于整体音量评估
   - RMS：用于能量水平控制
   - Max Amplitude：用于峰值控制

### 理想值参考
- **专业音频制作参考值**：
  - Mean Norm：0.1-0.3
  - RMS：0.1-0.4
  - Max Amplitude：0.8-0.9

### 使用建议
1. 先检查Max Amplitude避免削波
2. 用RMS确保整体能量适中
3. 参考Mean Norm调整整体音量
4. 三个指标要结合具体应用场景来判断

这些指标共同作用，帮助我们：
- 确保音频质量
- 保持音量一致性
- 避免失真和噪声
- 优化听感体验


**使用这个工具可以**：
- 获取详细的音量统计数据：
- 平均音量
- RMS音量
- 最大音量
- 最小音量
- 标准差等统计指标

**生成可视化报告**：
- 音量分布直方图
- 音量特征箱线图
- 音量相关性散点图

**输出建议**：

根据分析结果，你可以这样选择 target_db：      

1）如果想保持大多数音频的原有动态范围
- 设置 target_db = 平均音量的均值  

2）如果想确保所有音频都能清晰听到：   
- 设置 target_db = 平均音量均值 + 标准差

3）如果想避免音量过大：
- 设置 target_db = min(平均音量均值, -3.0)

## 如何使用
### 1、音频描述性分析

```python
from audio_analyzer import AudioAnalyzer
analyzer = AudioAnalyzer()

# 指定输入和输出目录
base_input_dir = "raw_voices"
analysis_output_dir = "raw_voices/音频分析报告"

# 执行分析
results_df = analyzer.analyze_speaker_directory(
    base_dir=base_input_dir,
    output_dir=analysis_output_dir,
    max_workers=16
)
```
```
发现 49 个说话人目录
处理说话人:   0%|          | 0/49 [00:00<?, ?it/s]

分析说话人: 廉颇

分析音频:   0%|          | 0/118 [00:00<?, ?it/s]
分析音频:  25%|██▌       | 30/118 [00:00<00:00, 289.97it/s]
分析音频:  53%|█████▎    | 62/118 [00:00<00:00, 299.46it/s]
分析音频:  78%|███████▊  | 92/118 [00:00<00:00, 298.95it/s]
                                                           

音频分析报告 说话人: 廉颇:
--------------------------------------------------
分析的音频文件总数: 118

音量统计:

Mean Norm:
  mean: 0.053
  std: 0.010
  min: 0.032
  max: 0.082

RMS Amplitude:
  mean: 0.089
  std: 0.015
  min: 0.057
  max: 0.131

Max Amplitude:
  mean: 0.546
  std: 0.123
  min: 0.293
  max: 0.882
处理说话人:   2%|▏         | 1/49 [00:01<01:03,  1.31s/it]

推荐的target_db值:
1. 保守设置 (保持动态范围): target_db = 0.053
2. 平衡设置 (确保清晰度): target_db = 0.063
3. 安全设置: target_db = -3.000

分析结果已保存到: raw_voices/音频分析报告/廉颇

分析说话人: 小乔

分析音频:   0%|          | 0/201 [00:00<?, ?it/s]
分析音频:  14%|█▍        | 28/201 [00:00<00:00, 268.48it/s]
分析音频:  29%|██▉       | 58/201 [00:00<00:00, 283.83it/s]
分析音频:  43%|████▎     | 87/201 [00:00<00:00, 281.59it/s]
分析音频:  60%|█████▉    | 120/201 [00:00<00:00, 297.76it/s]
分析音频:  75%|███████▍  | 150/201 [00:00<00:00, 294.95it/s]
分析音频:  90%|████████▉ | 180/201 [00:00<00:00, 289.50it/s]
                                                            

音频分析报告 说话人: 小乔:
--------------------------------------------------
分析的音频文件总数: 201

音量统计:

Mean Norm:
  mean: 0.052
  std: 0.019
  min: 0.012
  max: 0.135

RMS Amplitude:
  mean: 0.086
  std: 0.030
  min: 0.024
  max: 0.209

Max Amplitude:
  mean: 0.495
  std: 0.143
  min: 0.163
  max: 0.943
处理说话人:   4%|▍         | 2/49 [00:02<01:09,  1.49s/it]
...
```
**Results**
<img width="448" alt="image" src="https://github.com/user-attachments/assets/cafd0db8-d726-4c51-8e4f-68ab0deb0f7c" />



<img width="930" alt="image" src="https://github.com/user-attachments/assets/9c6841bc-a164-415a-83b3-a3f98818ac5b" />

### 2、音频归一化
```python
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tts_audio_normalizer import AudioProcessingParams, process_all_speakers

def process_all_speakers(
    base_input_dir: str,
    base_output_dir: str,
    params: Optional[AudioProcessingParams] = None,
    max_workers: int = 16
) -> Dict[str, Any]:
    """
    处理基础目录下所有说话人的音频文件
    """
    os.makedirs(base_output_dir, exist_ok=True)
    speaker_dirs = [d for d in Path(base_input_dir).iterdir() if d.is_dir()]
    normalizer = TTSAudioNormalizer()
    results = {}
    
    if params is None:
        params = AudioProcessingParams(
            output_format='wav',
            target_db=-3.0,
            rate=22050,
            channels=1,
            noise_reduction_enabled=True,
            noise_threshold_db=-30.0,
            equalizer_enabled=True,
            treble_gain=2.0,
            mid_gain=1.0,
            bass_gain=1.5
        )
    
    print(f"发现 {len(speaker_dirs)} 个说话人目录")
    
    # 使用tqdm包装说话人处理循环
    for speaker_dir in tqdm(speaker_dirs, desc="处理说话人", position=0):
        speaker_name = speaker_dir.name
        print(f"\n处理说话人: {speaker_name}")
        
        speaker_output_dir = os.path.join(base_output_dir, speaker_name)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        try:
            normalizer.batch_normalize_directory(
                input_dir=str(speaker_dir),
                output_dir=speaker_output_dir,
                params=params,
                max_workers=max_workers
            )
            
            quality_report = normalizer.get_quality_report()
            results[speaker_name] = {
                'status': 'success',
                'quality_metrics': quality_report
            }
            
        except Exception as e:
            print(f"处理说话人 {speaker_name} 时发生错误: {str(e)}")
            results[speaker_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # 打印总体处理结果
    print("\n处理完成统计:")
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"成功处理说话人数: {success_count}/{len(speaker_dirs)}")
    
    if success_count < len(speaker_dirs):
        print("\n处理失败的说话人:")
        for speaker, result in results.items():
            if result['status'] == 'failed':
                print(f"- {speaker}: {result['error']}")
    
    return results

# 使用示例：
if __name__ == "__main__":
    base_input_dir = "./origin_audio_segments"
    base_output_dir = "./normalized_audio_segments"
    
    # 设置处理参数
    params = AudioProcessingParams(
        output_format='wav',
        target_db=-3.0,
        rate=22050,
        channels=1,
        noise_reduction_enabled=True,
        noise_threshold_db=-30.0,
        equalizer_enabled=True,
        treble_gain=2.0,
        mid_gain=1.0,
        bass_gain=1.5
    )
    
    # 执行批量处理
    results = process_all_speakers(
        base_input_dir=base_input_dir,
        base_output_dir=base_output_dir,
        params=params,
        max_workers=16
    )
```



