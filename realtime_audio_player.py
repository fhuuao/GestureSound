import pygame
import json
import time
import sys
import threading
import numpy as np
import math

class SimpleTonePlayer:
    """简单的连续音调播放器"""
    
    def __init__(self, frequency, sample_rate=22050, duration=10.0):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.duration = duration
        self.sound = None
        self.channel = None
        self.is_playing = False
        self.volume = 0.6  # 固定音量
        
        # 生成长时间的音频样本
        self._generate_long_tone()
    
    def _generate_long_tone(self):
        """生成长时间的连续音调"""
        # 生成足够长的音频（10秒循环）
        samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, samples, False)
        
        # 生成正弦波
        wave = np.sin(2 * np.pi * self.frequency * t) * self.volume
        
        # 添加很短的淡入淡出避免爆音
        fade_samples = int(0.01 * self.sample_rate)  # 10ms
        if len(wave) > 2 * fade_samples:
            wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
            wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # 转换为pygame音频格式
        wave_int16 = (wave * 32767).astype(np.int16)
        
        # 创建立体声
        stereo_wave = np.column_stack((wave_int16, wave_int16))
        
        # 创建pygame Sound对象
        self.sound = pygame.sndarray.make_sound(stereo_wave)
    
    def start_playing(self):
        """开始播放"""
        if not self.is_playing and self.sound:
            self.channel = self.sound.play(loops=-1)  # 无限循环
            if self.channel:
                self.is_playing = True
                return True
        return False
    
    def stop_playing(self):
        """停止播放"""
        if self.is_playing and self.channel:
            self.channel.stop()
            self.channel = None
            self.is_playing = False
            return True
        return False
    
    def is_currently_playing(self):
        """检查是否正在播放"""
        if self.channel:
            return self.channel.get_busy()
        return False

class SimpleAudioPlayer:
    """简化的音频播放器"""
    
    def __init__(self):
        # 手指频率映射
        self.finger_frequencies = {
            "thumb": 261.63,   # do (C)
            "index": 293.66,   # re (D)
            "middle": 329.63,  # mi (E)
            "ring": 392.00,    # sol (G)
            "pinky": 440.00    # la (A)
        }
        
        # 手指状态
        self.finger_states = {
            "thumb": {"bent": False, "last_change": 0},
            "index": {"bent": False, "last_change": 0},
            "middle": {"bent": False, "last_change": 0},
            "ring": {"bent": False, "last_change": 0},
            "pinky": {"bent": False, "last_change": 0}
        }
        
        # 为每个手指创建音调播放器
        self.tone_players = {}
        
        # 防抖设置
        self.debounce_time = 100  # 毫秒
        
        # 初始化音频系统
        if not self.init_audio():
            raise Exception("音频系统初始化失败")
        
        # 创建音调播放器
        self._create_tone_players()
    
    def init_audio(self):
        """初始化pygame音频系统"""
        try:
            # 使用较低的采样率和较大的缓冲区以获得更稳定的播放
            pygame.mixer.pre_init(
                frequency=22050,   # 降低采样率
                size=-16,          # 16位有符号
                channels=2,        # 立体声
                buffer=2048        # 增大缓冲区
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)  # 支持多个同时播放
            
            print("✅ 音频系统初始化成功", file=sys.stderr)
            return True
        except pygame.error as e:
            print(f"❌ 音频系统初始化失败: {e}", file=sys.stderr)
            return False
    
    def _create_tone_players(self):
        """创建音调播放器"""
        print("🎵 创建音调播放器...", file=sys.stderr)
        for finger, freq in self.finger_frequencies.items():
            try:
                player = SimpleTonePlayer(freq)
                self.tone_players[finger] = player
                print(f"✅ 创建: {finger} -> {freq:.2f} Hz", file=sys.stderr)
            except Exception as e:
                print(f"❌ 创建失败 {finger}: {e}", file=sys.stderr)
    
    def start_finger_tone(self, finger_name):
        """开始播放手指音调"""
        if finger_name in self.tone_players:
            player = self.tone_players[finger_name]
            if player.start_playing():
                print(f"🎵 开始播放: {finger_name} ({self.finger_frequencies[finger_name]:.2f} Hz)", file=sys.stderr)
                return True
        return False
    
    def stop_finger_tone(self, finger_name):
        """停止播放手指音调"""
        if finger_name in self.tone_players:
            player = self.tone_players[finger_name]
            if player.stop_playing():
                print(f"⏹️ 停止播放: {finger_name}", file=sys.stderr)
                return True
        return False
    
    def process_gesture_data(self, gesture_data):
        """处理手势数据"""
        current_time = time.time() * 1000
        
        try:
            if isinstance(gesture_data, str):
                data = json.loads(gesture_data.strip())
            else:
                data = gesture_data
            
            finger_states = data.get("states", {})
            
            for finger_name in self.finger_states.keys():
                if finger_name in finger_states:
                    is_bent = finger_states[finger_name]
                    
                    finger_info = self.finger_states[finger_name]
                    last_bent = finger_info["bent"]
                    last_change_time = finger_info["last_change"]
                    
                    state_changed = (is_bent != last_bent)
                    time_since_change = current_time - last_change_time
                    debounce_ok = time_since_change > self.debounce_time
                    
                    if state_changed and debounce_ok:
                        if is_bent:
                            # 开始播放
                            self.start_finger_tone(finger_name)
                        else:
                            # 停止播放
                            self.stop_finger_tone(finger_name)
                        
                        finger_info["bent"] = is_bent
                        finger_info["last_change"] = current_time
                    
                    # 检查播放状态并修复断开的连接
                    if finger_info["bent"] and finger_name in self.tone_players:
                        player = self.tone_players[finger_name]
                        if not player.is_currently_playing():
                            # 如果应该播放但没有播放，重新启动
                            print(f"🔄 重新启动播放: {finger_name}", file=sys.stderr)
                            player.start_playing()
                            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ 处理手势数据出错: {e}", file=sys.stderr)
    
    def stop_all(self):
        """停止所有播放"""
        for finger_name in self.tone_players.keys():
            self.stop_finger_tone(finger_name)
    
    def run(self):
        """运行音频播放器主循环"""
        print("🎧 简化音频播放器已启动", file=sys.stderr)
        print("💡 等待手势数据...", file=sys.stderr)
        print("🎹 大拇指=do, 食指=re, 中指=mi, 无名指=sol, 小指=la", file=sys.stderr)
        print("🔄 手指弯曲时播放连续音调，伸直时停止", file=sys.stderr)
        print("🎵 固定音量，无音量变化", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        
        try:
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if line:
                        self.process_gesture_data(line)
                        
                except KeyboardInterrupt:
                    print("\n⏹️ 收到中断信号", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"❌ 读取输入出错: {e}", file=sys.stderr)
                    
        except Exception as e:
            print(f"❌ 播放器运行出错: {e}", file=sys.stderr)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 正在清理资源...", file=sys.stderr)
        self.stop_all()
        
        try:
            pygame.mixer.quit()
            print("🎵 音频系统已关闭", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}", file=sys.stderr)

def main():
    """主函数"""
    try:
        player = SimpleAudioPlayer()
        player.run()
    except Exception as e:
        print(f"❌ 播放器启动失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()