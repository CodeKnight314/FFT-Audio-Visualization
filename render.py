import librosa
import numpy as np
import os
from fft_ops import FFT_1D_optimized, padArray
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
from glob import glob 
import cv2 

def load_audio(path: str):
    """
    Loads audio file and returns the signal, sampling rate and duration of audio file
    
    Args: 
        path (str): path to mp3 file 
    
    Returns: 
        signal (np.array): numpy array that captures the magnitude at each sample
        sr (int): sampling rate of the audio file
        duration (float): duration of the audio file in seconds
    """
    assert path.endswith(".mp3"), "[ERROR] file is not in mp3 format"
    
    signals, sr = librosa.load(path, sr=None)
    
    duration = librosa.get_duration(y=signals, sr=sr)
        
    return signals, sr, duration

def stft_Audio(signals: np.array, sr: int, fps: int):
    """
    Compute STFT with specified frames per second.
    
    Args:
        signals (np.array): Input audio signals
        sr (int): Sampling rate of the audio files
        fps (int): Desired frames per second
    
    Returns:
        magnitude (np.array): normalized magnitude of the audio signals where each row represents a segment
        frequency (np.array): frequency array
        
    """ 
    total_duration = signals.shape[0] / sr
    
    num_segments = int(total_duration * fps)
    
    segment_size = sr
    segment_step = int(signals.shape[0] / num_segments)
    
    stft_data = np.zeros((num_segments, segment_size // 2 + 1), dtype=complex) 
    
    for i in tqdm(range(num_segments)): 
        center = i * segment_step
        
        start = center - (segment_size // 2)
        end = start + segment_size
        
        if start < 0:
            segment = signals[:segment_size]
        elif end > signals.shape[0]:
            segment = signals[-segment_size:]
        else:
            segment = signals[start:end]
            
        fft_result = FFT_1D_optimized(padArray(segment))
        stft_data[i, :] = fft_result[: segment_size // 2 + 1]
    
    magnitude = np.abs(stft_data)
    
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    freq_axis = np.fft.fftfreq(segment_size, 1/sr)[:segment_size // 2 + 1]

    return magnitude, freq_axis

def compile2video(dir: str, video_name: str, fps: int):
    """
    Compiles a directory of png files into a single mp4 file with desired fps
    
    Arguments: 
        dir (str): directory containing all images for video compilation
        video_name (str): video name
        fps (int): desired fps for video
    """
    frames = sorted(glob(os.path.join(dir, "*.png")))
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in tqdm(frames, desc="Compiling video: "): 
        video.write(cv2.imread(frame))
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

def generate_frames(audio_file: str, output_directory: str, fps: int = 10):
    """
    Generates and stores fft frames as Magnitude vs Frequency 
    
    Arguments: 
        audio_file (str): path reference to the audio file
        output_directory (str): directory for outputing frames
        fps (int): determines necessary frames for desired fps given samples
    """
    os.makedirs(output_directory, exist_ok=True)
    
    signals, sr, duration = load_audio(audio_file)
    magnitude, frequency = stft_Audio(signals, sr, fps)
    
    frequency_khz = frequency / 1000
    
    max_magnitude = np.max(magnitude)
    
    for i in tqdm(range(len(magnitude)), desc="Generating frames"):
        plt.figure(figsize=(12, 6))
        
        plt.plot(frequency_khz, magnitude[i], 'b-', linewidth=1)
        
        plt.xlim(0, max(frequency_khz))
        plt.ylim(0, max_magnitude)
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Magnitude')
        plt.title(f'Audio Spectrum - Time: {i:.1f}s')
        
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_directory, f'frame_{i:04d}.png'), bbox_inches='tight', dpi=150)
        plt.close()
    
    compile2video(output_directory, "Output.mp4", fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D FFT frames from an audio file.")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file (MP3)")
    parser.add_argument("output_directory", type=str, help="Directory to save the output frames")
    parser.add_argument("fps", type=int, help="FPS for video rendering")
    args = parser.parse_args()

    generate_frames(args.audio_file, args.output_directory, args.fps)
    