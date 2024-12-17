

# TTSAudioNormalizer

TTSAudioNormalizer 是一个专业的 TTS 音频预处理工具，提供全面的音频分析和标准化处理功能。本工具旨在提升 TTS 训练数据质量，确保音频特征的一致性。

## 主要功能

### 1. 音频分析
- 生成详细的响度统计报告
- 提供音量分布可视化
- 输出参数优化建议
```python
from audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()
results = analyzer.analyze_speaker_directory(
    base_dir="raw_voices",
    output_dir="analysis_report",
    max_workers=16
)
```
#### Results:
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

推荐的target_db值:
1. 保守设置 (保持动态范围): target_db = 0.052
2. 平衡设置 (确保清晰度): target_db = 0.071
3. 安全设置: target_db = -3.000

分析结果已保存到: raw_voices/音频分析报告/小乔

分析说话人: 赵云

分析音频:   0%|          | 0/142 [00:00<?, ?it/s]
分析音频:  20%|█▉        | 28/142 [00:00<00:00, 270.67it/s]
分析音频:  42%|████▏     | 60/142 [00:00<00:00, 294.19it/s]
分析音频:  63%|██████▎   | 90/142 [00:00<00:00, 291.33it/s]
分析音频:  85%|████████▍ | 120/142 [00:00<00:00, 283.42it/s]
                                                            

音频分析报告 说话人: 赵云:
--------------------------------------------------
分析的音频文件总数: 142

音量统计:

Mean Norm:
  mean: 0.050
  std: 0.019
  min: 0.018
  max: 0.124

RMS Amplitude:
  mean: 0.089
  std: 0.031
  min: 0.039
  max: 0.193

Max Amplitude:
  mean: 0.603
  std: 0.182
  min: 0.339
  max: 1.000
处理说话人:   6%|▌         | 3/49 [00:04<01:06,  1.45s/it]

推荐的target_db值:
1. 保守设置 (保持动态范围): target_db = 0.050
2. 平衡设置 (确保清晰度): target_db = 0.070
3. 安全设置: target_db = -3.000

分析结果已保存到: raw_voices/音频分析报告/赵云
...
```
<img width="381" alt="image" src="https://github.com/user-attachments/assets/32beed78-d703-419a-a77c-a6aa232c9adc" />
<img width="933" alt="image" src="https://github.com/user-attachments/assets/7e3a0f6c-007a-409f-bc07-2c6f6a002a6f" />




### 2. 音频标准化
```python
from tts_audio_normalizer import AudioProcessingParams, process_all_speakers

params = AudioProcessingParams(
    target_db=-3.0,          # 目标音量
    rate=22050,             # 采样率
    channels=1,             # 单声道
    noise_reduction_enabled=True,
    equalizer_enabled=True,
    treble_gain=2.0,        # 高频增益
    mid_gain=1.0,           # 中频增益
    bass_gain=1.5           # 低频增益
)

results = process_all_speakers(
    base_input_dir="raw_audio",
    base_output_dir="normalized_audio",
    params=params
)
```

## 参数配置指南

### 1. 基础参数
```python
# 基础格式设置
rate: int = 44100            # 采样率
channels: int = 1            # 声道数
output_format: str = 'wav'   # 输出格式
target_db: float = -3.0      # 目标音量
```

### 2. 音质优化参数
```python
# 均衡器设置
equalizer_enabled: bool = True    # 启用均衡
treble_frequency: float = 3000.0  # 高频中心(2-8kHz)
mid_frequency: float = 1000.0     # 中频中心(250Hz-2kHz)
bass_frequency: float = 100.0     # 低频中心(80-250Hz)
```

### 3. 降噪参数
```python
# 噪声处理
subsonic_filter_enabled: bool = True  # 超低频过滤
compression_ratio: float = 2.5        # 压缩比
threshold_db: float = -15.0           # 噪声阈值
```

## 场景优化建议

### 1. 语音类型适配
| 语音类型 | 推荐参数 |
|---------|---------|
| 男声 | bass_gain=2.0, mid_frequency=1200Hz |
| 女声 | treble_gain=1.5, bass_gain=1.5 |
| 儿童声 | mid_gain=1.5, bass_gain=1.0 |

### 2. 压限器配置
| 压缩程度 | 参数组合 |
|---------|---------|
| 温和压缩 | threshold_db=-20, ratio=2, attack=0.3s |
| 中等压缩 | threshold_db=-25, ratio=3, attack=0.2s |
| 强烈压缩 | threshold_db=-30, ratio=4, attack=0.1s |

### 3. 均衡器配置
| 音质目标 | 参数组合 |
|---------|---------|
| 人声增强 | treble=2.0, bass=1.0 |
| 清晰度提升 | treble=3.0, bass=-1.0 |
| 温暖音色 | treble=-1.0, bass=2.0 |

## 使用注意事项

1. **音频特征保护**
- 避免过度处理导致失真
- 保持音素边界清晰度
- 维持语音自然韵律

2. **数据集适配**
- 根据说话人特点调整参数
- 考虑录音环境因素
- 保持处理一致性

3. **质量控制**
- 定期检查处理效果
- 监控异常样本
- 及时调整参数

## 最佳实践流程

1. 先进行音频分析
2. 根据分析报告选择参数
3. 小批量测试处理效果
4. 调整优化参数配置
5. 执行批量标准化处理
6. 验证处理结果质量

通过合理配置和使用本工具，可以显著提升 TTS 训练数据质量，为模型训练提供更好的基础数据支持。
