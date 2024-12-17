import os
import sox
import numpy as np
import pandas as pd
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from pathlib import Path

# 设置中文字体
# 请根据你的系统调整字体路径
font_path = './xingchenyudahai.ttf'  # 中文字体路径
font_prop = FontProperties(fname=font_path)



class AudioAnalyzer:
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
        
    def analyze_audio(self, file_path: str, speaker_name: str) -> Dict:
        """
        分析音频文件的音量特征
        
        Args:
            file_path: 音频文件路径
            speaker_name: 说话人名称
            
        Returns:
            Dict: 包含音频特征的字典
        """
        try:
            stats = sox.file_info.stat(file_path)
            
            return {
                'speaker': speaker_name,
                'file_name': os.path.basename(file_path),
                'mean_norm': float(stats['Mean    norm']),
                'mean_amplitude': float(stats['Mean    amplitude']),
                'rms_amplitude': float(stats['RMS     amplitude']),
                'max_amplitude': float(stats['Maximum amplitude']),
                'min_amplitude': float(stats['Minimum amplitude']),
                'duration': float(stats['Length (seconds)']),
                'volume_adjustment': float(stats['Volume adjustment'])
            }
            
        except Exception as e:
            print(f"分析文件出错 {file_path}: {str(e)}")
            return None

    def analyze_speaker_directory(self, 
                                base_dir: str, 
                                output_dir: str,
                                max_workers: int = 4) -> pd.DataFrame:
        """
        分析包含多个说话人子文件夹的目录
        """
        # 获取所有说话人目录
        speaker_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        all_results = []
        
        print(f"发现 {len(speaker_dirs)} 个说话人目录")
        
        # 为每个说话人创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个说话人的音频
        for speaker_dir in tqdm(speaker_dirs, desc="处理说话人"):
            speaker_name = speaker_dir.name
            print(f"\n分析说话人: {speaker_name}")
            
            # 获取该说话人的所有音频文件
            audio_files = []
            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if any(file.lower().endswith(fmt) for fmt in self.supported_formats):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                print(f"警告: 未在 {speaker_name} 目录下找到音频文件")
                continue
                
            # 使用线程池处理音频文件
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.analyze_audio, file, speaker_name) 
                    for file in audio_files
                ]
                
                # 使用tqdm显示处理进度
                for future in tqdm(futures, desc="分析音频", leave=False):
                    result = future.result()
                    if result:
                        all_results.append(result)
            
            # 创建说话人特定的DataFrame
            speaker_df = pd.DataFrame([r for r in all_results if r['speaker'] == speaker_name])
            
            if not speaker_df.empty:
                # 为每个说话人生成单独的报告
                speaker_output_dir = os.path.join(output_dir, speaker_name)
                os.makedirs(speaker_output_dir, exist_ok=True)
                self.generate_analysis_report(
                    speaker_df, 
                    speaker_output_dir,
                    title_prefix=f"说话人: {speaker_name}"
                )
        
        # 创建总体DataFrame并生成总体报告
        all_df = pd.DataFrame(all_results)
        if not all_df.empty:
            self.generate_analysis_report(
                all_df, 
                output_dir,
                title_prefix="所有说话人总体分析"
            )
            
            # 生成说话人间对比报告
            self.generate_speaker_comparison(all_df, output_dir)
            
        return all_df

    def generate_speaker_comparison(self, df: pd.DataFrame, output_dir: str):
        """
        生成说话人之间的对比报告
        """
        plt.figure(figsize=(15, 10))
        
        # 计算每个说话人的平均指标
        speaker_stats = df.groupby('speaker').agg({
            'mean_norm': 'mean',
            'rms_amplitude': 'mean',
            'max_amplitude': 'mean'
        }).round(3)
        
        # 绘制说话人对比图
        ax = speaker_stats.plot(kind='bar', figsize=(15, 6))
        plt.title('说话人音量特征对比', fontsize=30, pad=20, fontproperties=font_prop)
        plt.xlabel('说话人', fontsize=12, fontproperties=font_prop)
        plt.ylabel('振幅', fontsize=12, fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        plt.legend(title='Statistical Indicators')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存对比图
        plt.savefig(os.path.join(output_dir, 'speaker_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        
        # 保存统计数据
        speaker_stats.to_csv(os.path.join(output_dir, 'speaker_comparison.csv'))
        
        print("\n说话人间对比统计:")
        print(speaker_stats)

    def generate_analysis_report(self, df: pd.DataFrame, output_dir: str = None, title_prefix: str = ""):
        """
        Generate analysis report with statistics and visualizations

        Args:
            df: DataFrame containing audio analysis results
            output_dir: Directory to save results
            title_prefix: Prefix for plot titles (e.g. speaker name)
        """
        # Calculate statistics
        stats = {
            'Mean Norm': {
                'mean': df['mean_norm'].mean(),
                'std': df['mean_norm'].std(),
                'min': df['mean_norm'].min(),
                'max': df['mean_norm'].max(),
            },
            'RMS Amplitude': {
                'mean': df['rms_amplitude'].mean(),
                'std': df['rms_amplitude'].std(),
                'min': df['rms_amplitude'].min(),
                'max': df['rms_amplitude'].max(),
            },
            'Max Amplitude': {
                'mean': df['max_amplitude'].mean(),
                'std': df['max_amplitude'].std(),
                'min': df['max_amplitude'].min(),
            'max': df['max_amplitude'].max(),
            }
        }

        # Print report
        print(f"\n音频分析报告 {title_prefix}:")
        print("-" * 50)
        print(f"分析的音频文件总数: {len(df)}")
        print("\n音量统计:")
        for metric, values in stats.items():
            print(f"\n{metric}:")
            for stat, value in values.items():
                print(f"  {stat}: {value:.3f}")

        # Create visualizations
        plt.figure(figsize=(15, 10))

        # Volume distribution histogram
        plt.subplot(2, 2, 1)
        plt.hist(df['mean_norm'], bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
        plt.title(f'{title_prefix}\n平均归一化分布', fontsize=12, pad=10, fontproperties=font_prop)
        plt.xlabel('Mean Norm', fontsize=10, fontproperties=font_prop)
        plt.ylabel('Frequency', fontsize=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Volume boxplot
        plt.subplot(2, 2, 2)
        df.boxplot(column=['mean_norm', 'rms_amplitude', 'max_amplitude'],
               patch_artist=True,
               boxprops=dict(facecolor='#6896F3', alpha=0.7),
               medianprops=dict(color='red'),
                  
                  )
        plt.title(f'{title_prefix}\n音量特征箱型图', fontsize=12, pad=10, fontproperties=font_prop)
        plt.ylabel('Amplitude', fontsize=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(df['mean_norm'], df['max_amplitude'], 
                   alpha=0.6, color='#2986F3', edgecolor='white')
        plt.xlabel('Mean Norm', fontsize=10, fontproperties=font_prop)
        plt.ylabel('Maximum Amplitude', fontsize=10, fontproperties=font_prop)
        plt.title(f'{title_prefix}\n平均归一化 vs 最大振幅', fontsize=12, pad=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Add overall title
        plt.suptitle(f'{title_prefix} 音频音量分析', fontsize=14, y=1.02, fontproperties=font_prop)

        # Adjust layout
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'audio_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            df.to_csv(os.path.join(output_dir, 'audio_analysis.csv'))

            # Recommend target_db values
            mean_norm = df['mean_norm'].mean()
            std_norm = df['mean_norm'].std()

            print("\n推荐的target_db值:")
            print(f"1. 保守设置 (保持动态范围): target_db = {mean_norm:.3f}")
            print(f"2. 平衡设置 (确保清晰度): target_db = {(mean_norm + std_norm):.3f}")
            print(f"3. 安全设置: target_db = {min(mean_norm, -3.0):.3f}")

            print(f"\n分析结果已保存到: {output_dir}")
        else:
            plt.show()

        plt.close()  # 确保关闭图形，避免内存泄漏
