# five_tones.py
# 生成五个基本音调：do re mi sol la
# 按照手指映射：大拇指-do, 食指-re, 中指-mi, 无名指-sol, 小指-la

import numpy as np
from scipy.io import wavfile
import os

def generate_tone(frequency, duration=1.0, sample_rate=44100):
    """生成指定频率的正弦波音调，带淡入淡出效果"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成正弦波
    wave = np.sin(2 * np.pi * frequency * t)
    
    # 添加淡入淡出效果，避免爆音
    fade_frames = int(0.05 * sample_rate)  # 50ms淡入淡出
    
    # 淡入
    wave[:fade_frames] *= np.linspace(0, 1, fade_frames)
    # 淡出
    wave[-fade_frames:] *= np.linspace(1, 0, fade_frames)
    
    return wave

# 创建sounds文件夹
if not os.path.exists('sounds'):
    os.makedirs('sounds')
    print('创建了 sounds 文件夹')

# 手指对应的音调映射 (按照手势识别程序中的索引)
finger_notes = {
    'thumb': ('do', 261.63),    # 大拇指 - do (C)  -> thumb.wav
    'index': ('re', 293.66),    # 食指 - re (D)    -> index.wav  
    'middle': ('mi', 329.63),   # 中指 - mi (E)    -> middle.wav
    'ring': ('sol', 392.00),    # 无名指 - sol (G)  -> ring.wav
    'pinky': ('la', 440.00)     # 小指 - la (A)    -> pinky.wav
}

print('正在生成手指音调映射文件...')
print('大拇指: do, 食指: re, 中指: mi, 无名指: sol, 小指: la')
print('-' * 50)

# 生成每个手指对应的音调文件
for finger, (note, freq) in finger_notes.items():
    # 生成0.8秒的音调，比较适合手势触发
    wave = generate_tone(freq, duration=0.8)
    
    # 转换为16位整数格式保存
    wave_int16 = (wave * 32767 * 0.7).astype(np.int16)  # 降低音量到70%
    
    # 保存到sounds文件夹
    filename = f'sounds/{finger}.wav'
    wavfile.write(filename, 44100, wave_int16)
    print(f'✓ 已生成: {filename} -> {note} ({freq:.2f} Hz)')

# 额外生成原始的do re mi sol la文件（可选）
print('\n生成原始音调文件...')
original_notes = {
    'do': 261.63,   # C
    're': 293.66,   # D  
    'mi': 329.63,   # E
    'sol': 392.00,  # G
    'la': 440.00    # A
}

for note, freq in original_notes.items():
    wave = generate_tone(freq, duration=1.0)
    wave_int16 = (wave * 32767 * 0.7).astype(np.int16)
    wavfile.write(f'{note}.wav', 44100, wave_int16)
    print(f'✓ 已生成: {note}.wav ({freq:.2f} Hz)')

print('\n' + '='*50)
print('完成！生成的文件：')
print('手势识别专用文件（在sounds/文件夹中）：')
for finger, (note, freq) in finger_notes.items():
    print(f'  sounds/{finger}.wav -> {note}')
print('\n原始音调文件（在项目根目录）：')
for note in original_notes.keys():
    print(f'  {note}.wav')
print('\n现在可以运行手势识别程序了！')