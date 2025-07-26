import pygame
import json
import time
import sys
import threading
import numpy as np
import math
import traceback
import gc

class SimpleTonePlayer:
    """简单的连续音调播放器"""
    
    def __init__(self, frequency, sample_rate=22050, duration=5.0):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.duration = duration
        self.sound = None
        self.channel = None
        self.is_playing = False
        self.volume = 0.5  # 降低音量避免音频设备过载
        self.last_start_time = 0
        
        # 生成音频样本
        self._generate_tone()
    
    def _generate_tone(self):
        """生成音调"""
        try:
            # 生成较短的音频避免内存占用过大
            samples = int(self.sample_rate * self.duration)
            t = np.linspace(0, self.duration, samples, False)
            
            # 生成正弦波
            wave = np.sin(2 * np.pi * self.frequency * t) * self.volume
            
            # 添加淡入淡出
            fade_samples = int(0.02 * self.sample_rate)  # 20ms
            if len(wave) > 2 * fade_samples:
                wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
                wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # 转换格式
            wave_int16 = (wave * 32767).astype(np.int16)
            stereo_wave = np.column_stack((wave_int16, wave_int16))
            
            # 创建pygame Sound对象
            self.sound = pygame.sndarray.make_sound(stereo_wave)
            
        except Exception as e:
            print(f"❌ 生成音调失败 {self.frequency}Hz: {e}", file=sys.stderr)
            self.sound = None
    
    def start_playing(self):
        """开始播放"""
        try:
            current_time = time.time()
            # 防止过于频繁的启动请求
            if current_time - self.last_start_time < 0.1:
                return self.is_playing
            
            if not self.is_playing and self.sound:
                self.channel = self.sound.play(loops=-1)
                if self.channel:
                    self.is_playing = True
                    self.last_start_time = current_time
                    return True
            return self.is_playing
        except Exception as e:
            print(f"❌ 播放启动失败 {self.frequency}Hz: {e}", file=sys.stderr)
            return False
    
    def stop_playing(self):
        """停止播放"""
        try:
            if self.is_playing and self.channel:
                self.channel.stop()
                self.channel = None
                self.is_playing = False
                return True
            return False
        except Exception as e:
            print(f"❌ 停止播放失败 {self.frequency}Hz: {e}", file=sys.stderr)
            return False
    
    def is_currently_playing(self):
        """检查是否正在播放"""
        try:
            if self.channel:
                is_busy = self.channel.get_busy()
                if not is_busy and self.is_playing:
                    # 同步状态
                    self.is_playing = False
                    self.channel = None
                return is_busy
            return False
        except Exception as e:
            print(f"❌ 检查播放状态失败 {self.frequency}Hz: {e}", file=sys.stderr)
            self.is_playing = False
            self.channel = None
            return False

class RobustAudioPlayer:
    """增强稳定性的音频播放器"""
    
    def __init__(self):
        # 手指频率映射
        self.finger_frequencies = {
            "thumb": 261.63,   # do (C)
            "index": 293.66,   # re (D)
            "middle": 329.63,  # mi (E)
            "ring": 392.00,    # sol (G)
            "pinky": 440.00    # la (A)
        }
        
        # 手指状态跟踪
        self.finger_states = {
            "thumb": {"bent": False, "last_change": 0, "playing": False, "error_count": 0},
            "index": {"bent": False, "last_change": 0, "playing": False, "error_count": 0},
            "middle": {"bent": False, "last_change": 0, "playing": False, "error_count": 0},
            "ring": {"bent": False, "last_change": 0, "playing": False, "error_count": 0},
            "pinky": {"bent": False, "last_change": 0, "playing": False, "error_count": 0}
        }
        
        # 音调播放器
        self.tone_players = {}
        
        # 防抖和错误控制
        self.debounce_time = 120  # 毫秒，增加防抖时间
        self.max_errors_per_finger = 10
        
        # 统计和监控
        self.stats = {
            "messages_received": 0,
            "messages_per_second": 0,
            "state_changes": 0,
            "audio_starts": 0,
            "audio_stops": 0,
            "errors": 0,
            "last_stats_time": time.time(),
            "last_message_time": time.time()
        }
        
        # 系统状态
        self.system_running = True
        self.last_gc_time = time.time()
        self.last_heartbeat = time.time()
        
        # 初始化音频系统
        if not self.init_audio():
            raise Exception("音频系统初始化失败")
        
        # 创建音调播放器
        self._create_tone_players()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def init_audio(self):
        """初始化pygame音频系统"""
        try:
            # 更保守的音频设置
            pygame.mixer.pre_init(
                frequency=22050,
                size=-16,
                channels=2,
                buffer=4096  # 更大的缓冲区提高稳定性
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(6)  # 限制通道数
            
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
                if player.sound:  # 只添加成功创建的播放器
                    self.tone_players[finger] = player
                    print(f"✅ 创建: {finger} -> {freq:.2f} Hz", file=sys.stderr)
                else:
                    print(f"❌ 创建失败: {finger} -> {freq:.2f} Hz", file=sys.stderr)
            except Exception as e:
                print(f"❌ 创建异常 {finger}: {e}", file=sys.stderr)
    
    def _monitor_system(self):
        """系统监控线程"""
        while self.system_running:
            try:
                current_time = time.time()
                
                # 心跳检测
                if current_time - self.last_heartbeat > 15:
                    self._print_heartbeat()
                    self.last_heartbeat = current_time
                
                # 定期垃圾回收
                if current_time - self.last_gc_time > 60:
                    gc.collect()
                    self.last_gc_time = current_time
                    print("🧹 执行垃圾回收", file=sys.stderr)
                
                # 检查长时间无消息
                if current_time - self.stats["last_message_time"] > 30:
                    print("⚠️ 长时间未收到消息，可能连接中断", file=sys.stderr)
                
                # 检查播放器状态并修复
                self._check_and_repair_players()
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                print(f"❌ 监控线程错误: {e}", file=sys.stderr)
                time.sleep(10)
    
    def _check_and_repair_players(self):
        """检查并修复播放器状态"""
        try:
            for finger_name, finger_info in self.finger_states.items():
                if finger_name in self.tone_players:
                    player = self.tone_players[finger_name]
                    
                    # 如果应该播放但没有播放，尝试修复
                    if finger_info["bent"] and finger_info["playing"]:
                        if not player.is_currently_playing():
                            print(f"🔧 修复播放状态: {finger_name}", file=sys.stderr)
                            if player.start_playing():
                                finger_info["error_count"] = 0
                            else:
                                finger_info["error_count"] += 1
                                if finger_info["error_count"] > self.max_errors_per_finger:
                                    print(f"❌ {finger_name} 错误过多，停止播放", file=sys.stderr)
                                    finger_info["playing"] = False
                                    finger_info["bent"] = False
        except Exception as e:
            print(f"❌ 修复播放器时出错: {e}", file=sys.stderr)
    
    def _print_heartbeat(self):
        """打印心跳信息"""
        try:
            playing_count = sum(1 for info in self.finger_states.values() if info["playing"])
            print(f"💗 音频播放器心跳: 接收={self.stats['messages_received']}, "
                  f"播放={playing_count}/5, 错误={self.stats['errors']}", file=sys.stderr)
            
            # 重置消息计数
            self.stats['messages_received'] = 0
            self.stats['errors'] = 0
        except Exception as e:
            print(f"❌ 心跳打印错误: {e}", file=sys.stderr)
    
    def start_finger_tone(self, finger_name):
        """开始播放手指音调"""
        try:
            if finger_name in self.tone_players:
                player = self.tone_players[finger_name]
                if player.start_playing():
                    self.finger_states[finger_name]["playing"] = True
                    self.finger_states[finger_name]["error_count"] = 0
                    self.stats["audio_starts"] += 1
                    print(f"🎵 开始播放: {finger_name} ({self.finger_frequencies[finger_name]:.2f} Hz)", file=sys.stderr)
                    return True
                else:
                    self.finger_states[finger_name]["error_count"] += 1
                    self.stats["errors"] += 1
            return False
        except Exception as e:
            print(f"❌ 启动播放失败 {finger_name}: {e}", file=sys.stderr)
            self.stats["errors"] += 1
            return False
    
    def stop_finger_tone(self, finger_name):
        """停止播放手指音调"""
        try:
            if finger_name in self.tone_players:
                player = self.tone_players[finger_name]
                if player.stop_playing():
                    self.finger_states[finger_name]["playing"] = False
                    self.finger_states[finger_name]["error_count"] = 0
                    self.stats["audio_stops"] += 1
                    print(f"⏹️ 停止播放: {finger_name}", file=sys.stderr)
                    return True
                else:
                    self.finger_states[finger_name]["error_count"] += 1
                    self.stats["errors"] += 1
            return False
        except Exception as e:
            print(f"❌ 停止播放失败 {finger_name}: {e}", file=sys.stderr)
            self.stats["errors"] += 1
            return False
    
    def process_states_data(self, states_data):
        """处理手势状态数据"""
        try:
            current_time = time.time() * 1000
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = time.time()
            
            # 解析数据
            if isinstance(states_data, str):
                finger_states = json.loads(states_data.strip())
            else:
                finger_states = states_data
            
            # 处理每个手指
            for finger_name in self.finger_states.keys():
                if finger_name in finger_states:
                    new_bent_state = finger_states[finger_name]
                    
                    finger_info = self.finger_states[finger_name]
                    old_bent_state = finger_info["bent"]
                    last_change_time = finger_info["last_change"]
                    
                    # 检查状态变化和防抖
                    state_changed = (new_bent_state != old_bent_state)
                    time_since_change = current_time - last_change_time
                    debounce_ok = time_since_change > self.debounce_time
                    
                    # 跳过错误过多的手指
                    if finger_info["error_count"] > self.max_errors_per_finger:
                        continue
                    
                    if state_changed and debounce_ok:
                        if new_bent_state:
                            # 开始播放
                            if self.start_finger_tone(finger_name):
                                finger_info["bent"] = new_bent_state
                                finger_info["last_change"] = current_time
                                self.stats["state_changes"] += 1
                        else:
                            # 停止播放
                            if self.stop_finger_tone(finger_name):
                                finger_info["bent"] = new_bent_state
                                finger_info["last_change"] = current_time
                                self.stats["state_changes"] += 1
                            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}", file=sys.stderr)
            self.stats["errors"] += 1
        except Exception as e:
            print(f"❌ 处理数据错误: {e}", file=sys.stderr)
            self.stats["errors"] += 1
    
    def stop_all(self):
        """停止所有播放"""
        try:
            for finger_name in list(self.tone_players.keys()):
                self.stop_finger_tone(finger_name)
            print("⏹️ 停止所有播放", file=sys.stderr)
        except Exception as e:
            print(f"❌ 停止所有播放时出错: {e}", file=sys.stderr)
    
    def run(self):
        """运行音频播放器主循环"""
        print("🎧 增强稳定性音频播放器已启动", file=sys.stderr)
        print("📊 接收手指状态数据 (bent: true/false)", file=sys.stderr)
        print("⚡ 高频数据接收模式，增强错误处理", file=sys.stderr)
        print("💡 等待状态数据...", file=sys.stderr)
        print("🎹 手指映射: 大拇指=do, 食指=re, 中指=mi, 无名指=sol, 小指=la", file=sys.stderr)
        print("🔄 手指弯曲=播放, 手指伸直=停止", file=sys.stderr)
        print(f"⏱️ 防抖时间: {self.debounce_time}ms", file=sys.stderr)
        print("🛡️ 增强稳定性和错误恢复", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        
        consecutive_errors = 0
        max_consecutive_errors = 20
        
        try:
            while self.system_running:
                try:
                    # 设置超时读取
                    sys.stdin.settimeout(1.0) if hasattr(sys.stdin, 'settimeout') else None
                    
                    line = sys.stdin.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if line:
                        self.process_states_data(line)
                        consecutive_errors = 0  # 重置连续错误计数
                        
                except EOFError:
                    print("📡 输入流结束", file=sys.stderr)
                    break
                except KeyboardInterrupt:
                    print("\n⏹️ 收到中断信号", file=sys.stderr)
                    break
                except Exception as e:
                    consecutive_errors += 1
                    print(f"❌ 读取输入错误 ({consecutive_errors}): {e}", file=sys.stderr)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("❌ 连续错误过多，退出程序", file=sys.stderr)
                        break
                    
                    time.sleep(0.1)  # 错误后短暂等待
                    
        except Exception as e:
            print(f"❌ 播放器主循环严重错误: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 正在清理音频播放器资源...", file=sys.stderr)
        
        # 停止监控线程
        self.system_running = False
        
        # 停止所有播放
        self.stop_all()
        
        # 等待监控线程结束
        try:
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=2)
        except:
            pass
        
        # 打印最终统计
        try:
            playing_count = sum(1 for info in self.finger_states.values() if info["playing"])
            print(f"📈 最终统计: 状态变化={self.stats['state_changes']}, "
                  f"播放={self.stats['audio_starts']}, 停止={self.stats['audio_stops']}, "
                  f"错误={self.stats['errors']}", file=sys.stderr)
        except:
            pass
        
        # 关闭音频系统
        try:
            pygame.mixer.quit()
            print("🎵 音频系统已关闭", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ 关闭音频系统时出错: {e}", file=sys.stderr)
        
        # 最终垃圾回收
        gc.collect()

def main():
    """主函数"""
    try:
        player = RobustAudioPlayer()
        player.run()
    except Exception as e:
        print(f"❌ 播放器启动失败: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()