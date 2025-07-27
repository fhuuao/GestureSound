import pygame
import json
import sys
import numpy as np
import traceback
import time

class FixedAudioPlayer:
    def __init__(self):
        # 手指对应的音频频率
        self.frequencies = {
            "thumb": 261.63,   # do (C)
            "index": 293.66,   # re (D)
            "middle": 329.63,  # mi (E)
            "ring": 392.00,    # sol (G)
            "pinky": 440.00    # la (A)
        }
        
        # 当前播放状态
        self.playing = {finger: False for finger in self.frequencies}
        self.channels = {}
        
        # 防抖设置
        self.last_change = {finger: 0 for finger in self.frequencies}
        self.debounce_time = 100  # 100ms防抖
        
        # 初始化音频
        if not self._init_audio():
            raise Exception("音频初始化失败")
        
        # 创建音频（短音调，不使用无限循环）
        self.sounds = {}
        for finger, freq in self.frequencies.items():
            self.sounds[finger] = self._create_tone(freq)
    
    def _init_audio(self):
        """初始化音频系统"""
        configs = [
            {"frequency": 22050, "size": -16, "channels": 2, "buffer": 1024},
            {"frequency": 22050, "size": -16, "channels": 1, "buffer": 512},
            {"frequency": 11025, "size": -16, "channels": 2, "buffer": 512},
            {},  # 默认配置
        ]
        
        for i, config in enumerate(configs):
            try:
                pygame.mixer.quit()  # 清理之前的初始化
                time.sleep(0.1)  # 给系统时间清理
                
                if config:
                    pygame.mixer.pre_init(**config)
                
                pygame.mixer.init()
                pygame.mixer.set_num_channels(8)  # 增加通道数
                
                # 测试基本功能
                freq, size, channels = pygame.mixer.get_init()
                print(f"✅ 音频初始化成功: {freq}Hz, {size}bit, {channels}ch", file=sys.stderr)
                return True
                
            except Exception as e:
                print(f"⚠️ 音频配置{i+1}失败: {e}", file=sys.stderr)
                continue
        
        print("❌ 所有音频配置都失败", file=sys.stderr)
        return False
    
    def _create_tone(self, frequency, duration=0.5):
        """创建短音调，避免无限循环"""
        try:
            # 获取当前音频设置
            freq, size, channels = pygame.mixer.get_init()
            sample_rate = freq
            samples = int(sample_rate * duration)
            
            # 生成正弦波
            t = np.linspace(0, duration, samples, False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # 添加淡入淡出
            fade_len = int(0.02 * sample_rate)  # 20ms淡入淡出
            if len(wave) > 2 * fade_len:
                wave[:fade_len] *= np.linspace(0, 1, fade_len)
                wave[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            # 转换为pygame格式
            wave_int16 = (wave * 16383).astype(np.int16)
            
            if channels == 2:
                stereo_wave = np.column_stack((wave_int16, wave_int16))
            else:
                stereo_wave = wave_int16
            
            return pygame.sndarray.make_sound(stereo_wave)
        except Exception as e:
            print(f"❌ 创建音调失败 {frequency}Hz: {e}", file=sys.stderr)
            return None
    
    def play_finger(self, finger):
        """播放手指音调（短音调，连续触发）"""
        try:
            current_time = time.time() * 1000
            
            # 防抖检查
            if current_time - self.last_change[finger] < self.debounce_time:
                return
            
            if finger in self.sounds and self.sounds[finger]:
                # 停止之前的播放
                if finger in self.channels and self.channels[finger]:
                    try:
                        self.channels[finger].stop()
                    except:
                        pass
                
                # 播放新的音调（不使用无限循环）
                channel = self.sounds[finger].play()
                if channel:
                    self.channels[finger] = channel
                    self.playing[finger] = True
                    self.last_change[finger] = current_time
                    print(f"🎵 {finger}", file=sys.stderr)
                    
        except Exception as e:
            print(f"❌ 播放失败 {finger}: {e}", file=sys.stderr)
    
    def stop_finger(self, finger):
        """停止手指音调"""
        try:
            current_time = time.time() * 1000
            
            # 防抖检查
            if current_time - self.last_change[finger] < self.debounce_time:
                return
                
            if finger in self.channels and self.playing[finger]:
                try:
                    if self.channels[finger]:
                        self.channels[finger].stop()
                except:
                    pass
                
                self.playing[finger] = False
                self.last_change[finger] = current_time
                print(f"⏹️ {finger}", file=sys.stderr)
                
        except Exception as e:
            print(f"❌ 停止失败 {finger}: {e}", file=sys.stderr)
    
    def process_data(self, data):
        """处理接收到的数据"""
        try:
            states = json.loads(data)
            
            for finger in self.frequencies:
                if finger in states:
                    if states[finger]:  # 弯曲
                        if not self.playing[finger]:  # 只有在没有播放时才开始播放
                            self.play_finger(finger)
                    else:  # 伸直
                        if self.playing[finger]:  # 只有在播放时才停止
                            self.stop_finger(finger)
                        
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ 数据处理错误: {e}", file=sys.stderr)
    
    def stop_all(self):
        """停止所有播放"""
        for finger in self.frequencies:
            if self.playing[finger]:
                try:
                    if finger in self.channels and self.channels[finger]:
                        self.channels[finger].stop()
                    self.playing[finger] = False
                except Exception as e:
                    print(f"⚠️ 停止{finger}时出错: {e}", file=sys.stderr)
    
    def cleanup_dead_channels(self):
        """清理已结束的通道"""
        try:
            for finger in list(self.channels.keys()):
                if finger in self.channels and self.channels[finger]:
                    if not self.channels[finger].get_busy():
                        self.playing[finger] = False
                        del self.channels[finger]
        except Exception as e:
            print(f"⚠️ 清理通道时出错: {e}", file=sys.stderr)
    
    def run(self):
        """主循环"""
        print("🎧 修复版音频播放器启动", file=sys.stderr)
        print("💡 等待手指状态数据...", file=sys.stderr)
        print("🎹 频率映射: thumb=do, index=re, middle=mi, ring=sol, pinky=la", file=sys.stderr)
        print("🔧 使用短音调避免音频卡住", file=sys.stderr)
        
        message_count = 0
        last_cleanup = time.time()
        
        try:
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if line:
                        self.process_data(line)
                        message_count += 1
                        
                        # 定期清理已结束的通道
                        current_time = time.time()
                        if current_time - last_cleanup > 1.0:  # 每秒清理一次
                            self.cleanup_dead_channels()
                            last_cleanup = current_time
                            
                            # 每1000条消息打印一次状态
                            if message_count % 1000 == 0:
                                playing_count = sum(1 for p in self.playing.values() if p)
                                print(f"📊 处理消息: {message_count}, 播放中: {playing_count}", file=sys.stderr)
                    
                except Exception as e:
                    print(f"❌ 处理输入错误: {e}", file=sys.stderr)
                    time.sleep(0.01)  # 短暂暂停避免疯狂循环
                    
        except KeyboardInterrupt:
            print("\n🛑 音频播放器收到中断信号", file=sys.stderr)
        except Exception as e:
            print(f"❌ 主循环错误: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理音频播放器...", file=sys.stderr)
        self.stop_all()
        
        try:
            # 等待所有声音停止
            time.sleep(0.1)
            pygame.mixer.quit()
            print("🎵 音频系统已关闭", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ 清理时出错: {e}", file=sys.stderr)

def main():
    """主函数"""
    try:
        player = FixedAudioPlayer()
        player.run()
    except Exception as e:
        print(f"❌ 音频播放器启动失败: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 音频播放器被中断", file=sys.stderr)
    except Exception as e:
        print(f"❌ 程序异常: {e}", file=sys.stderr)
    finally:
        try:
            pygame.mixer.quit()
        except:
            pass
        sys.exit(0)