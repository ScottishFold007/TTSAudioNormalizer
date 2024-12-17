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

# Set Chinese font
# Please adjust the font path according to your system
font_path = './xingchenyudahai.ttf'  # Chinese font path
font_prop = FontProperties(fname=font_path)

class AudioAnalyzer:
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a']
        
    def analyze_audio(self, file_path: str, speaker_name: str) -> Dict:
        """
        Analyze volume characteristics of audio file
        
        Args:
            file_path: Path to audio file
            speaker_name: Name of the speaker
            
        Returns:
            Dict: Dictionary containing audio features
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
            print(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def analyze_speaker_directory(self, 
                                base_dir: str, 
                                output_dir: str,
                                max_workers: int = 4) -> pd.DataFrame:
        """
        Analyze directory containing multiple speaker subdirectories
        """
        # Get all speaker directories
        speaker_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        all_results = []
        
        print(f"Found {len(speaker_dirs)} speaker directories")
        
        # Create output directory for each speaker
        os.makedirs(output_dir, exist_ok=True)
        
        # Process audio for each speaker
        for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
            speaker_name = speaker_dir.name
            print(f"\nAnalyzing speaker: {speaker_name}")
            
            # Get all audio files for this speaker
            audio_files = []
            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if any(file.lower().endswith(fmt) for fmt in self.supported_formats):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                print(f"Warning: No audio files found in {speaker_name} directory")
                continue
                
            # Use thread pool to process audio files
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.analyze_audio, file, speaker_name) 
                    for file in audio_files
                ]
                
                # Show processing progress with tqdm
                for future in tqdm(futures, desc="Analyzing audio", leave=False):
                    result = future.result()
                    if result:
                        all_results.append(result)
            
            # Create speaker-specific DataFrame
            speaker_df = pd.DataFrame([r for r in all_results if r['speaker'] == speaker_name])
            
            if not speaker_df.empty:
                # Generate individual report for each speaker
                speaker_output_dir = os.path.join(output_dir, speaker_name)
                os.makedirs(speaker_output_dir, exist_ok=True)
                self.generate_analysis_report(
                    speaker_df, 
                    speaker_output_dir,
                    title_prefix=f"Speaker: {speaker_name}"
                )
        
        # Create overall DataFrame and generate overall report
        all_df = pd.DataFrame(all_results)
        if not all_df.empty:
            self.generate_analysis_report(
                all_df, 
                output_dir,
                title_prefix="Overall Analysis of All Speakers"
            )
            
            # Generate speaker comparison report
            self.generate_speaker_comparison(all_df, output_dir)
            
        return all_df

    def generate_speaker_comparison(self, df: pd.DataFrame, output_dir: str):
        """
        Generate comparison report between speakers
        """
        plt.figure(figsize=(15, 10))
        
        # Calculate average metrics for each speaker
        speaker_stats = df.groupby('speaker').agg({
            'mean_norm': 'mean',
            'rms_amplitude': 'mean',
            'max_amplitude': 'mean'
        }).round(3)
        
        # Plot speaker comparison chart
        ax = speaker_stats.plot(kind='bar', figsize=(15, 6))
        plt.title('Speaker Volume Characteristics Comparison', fontsize=30, pad=20, fontproperties=font_prop)
        plt.xlabel('Speaker', fontsize=12, fontproperties=font_prop)
        plt.ylabel('Amplitude', fontsize=12, fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        plt.legend(title='Statistical Indicators')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save comparison chart
        plt.savefig(os.path.join(output_dir, 'speaker_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Save statistics
        speaker_stats.to_csv(os.path.join(output_dir, 'speaker_comparison.csv'))
        
        print("\nSpeaker Comparison Statistics:")
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
        print(f"\nAudio Analysis Report {title_prefix}:")
        print("-" * 50)
        print(f"Total audio files analyzed: {len(df)}")
        print("\nVolume statistics:")
        for metric, values in stats.items():
            print(f"\n{metric}:")
            for stat, value in values.items():
                print(f"  {stat}: {value:.3f}")

        # Create visualizations
        plt.figure(figsize=(15, 10))

        # Volume distribution histogram
        plt.subplot(2, 2, 1)
        plt.hist(df['mean_norm'], bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
        plt.title(f'{title_prefix}\nMean Normalization Distribution', fontsize=12, pad=10, fontproperties=font_prop)
        plt.xlabel('Mean Norm', fontsize=10, fontproperties=font_prop)
        plt.ylabel('Frequency', fontsize=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Volume boxplot
        plt.subplot(2, 2, 2)
        df.boxplot(column=['mean_norm', 'rms_amplitude', 'max_amplitude'],
               patch_artist=True,
               boxprops=dict(facecolor='#6896F3', alpha=0.7),
               medianprops=dict(color='red'))
        plt.title(f'{title_prefix}\nVolume Features Boxplot', fontsize=12, pad=10, fontproperties=font_prop)
        plt.ylabel('Amplitude', fontsize=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(df['mean_norm'], df['max_amplitude'], 
                   alpha=0.6, color='#2986F3', edgecolor='white')
        plt.xlabel('Mean Norm', fontsize=10, fontproperties=font_prop)
        plt.ylabel('Maximum Amplitude', fontsize=10, fontproperties=font_prop)
        plt.title(f'{title_prefix}\nMean Norm vs Maximum Amplitude', fontsize=12, pad=10, fontproperties=font_prop)
        plt.grid(True, alpha=0.3)

        # Add overall title
        plt.suptitle(f'{title_prefix} Audio Volume Analysis', fontsize=14, y=1.02, fontproperties=font_prop)

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

            print("\nRecommended target_db values:")
            print(f"1. Conservative setting (maintain dynamic range): target_db = {mean_norm:.3f}")
            print(f"2. Balanced setting (ensure clarity): target_db = {(mean_norm + std_norm):.3f}")
            print(f"3. Safe setting: target_db = {min(mean_norm, -3.0):.3f}")

            print(f"\nAnalysis results saved to: {output_dir}")
        else:
            plt.show()

        plt.close()  # Close the figure to avoid memory leaks
