import pygame
import json
import time
import sys
import os

class GestureAudioPlayer:
    """手势音频播放器"""
    
    def __init__(self, sounds_dir="sounds"):
        self.sounds_dir = sounds_dir
        self.sounds = {}
        
        # 手指状态跟踪
        self.finger_states = {
            "thumb": {"bent": False, "last_change": 0},
            "index": {"bent": False, "last_change": 0},
            "middle": {"bent": False, "last_change": 0},
            "ring": {"bent": False, "last_change": 0},
            "pinky": {"bent": False, "last_change": 0}
        }
        
        # 防抖设置（毫秒）
        self.debounce_time = 200
        
        # 初始化音频系统
        if not self.init_audio():
            sys.exit(1)
        
        # 加载音频文件
        if not self.load_sounds():
            sys.exit(1)
    
    def init_audio(self):
        """初始化pygame音频系统"""
        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)  # 支持多手指同时播放
            print("✅ 音频系统初始化成功", file=sys.stderr)
            return True
        except pygame.error as e:
            print(f"❌ 音频系统初始化失败: {e}", file=sys.stderr)
            return False
    
    def load_sounds(self):
        """加载所有音频文件"""
        finger_files = {
            "thumb": "thumb.wav",
            "index": "index.wav", 
            "middle": "middle.wav",
            "ring": "ring.wav",
            "pinky": "pinky.wav"
        }
        
        print(f"📂 从 {self.sounds_dir} 加载音频文件...", file=sys.stderr)
        
        loaded_count = 0
        for finger, filename in finger_files.items():
            filepath = os.path.join(self.sounds_dir, filename)
            if os.path.exists(filepath):
                try:
                    sound = pygame.mixer.Sound(filepath)
                    self.sounds[finger] = sound
                    print(f"✅ 加载: {finger} -> {filepath}", file=sys.stderr)
                    loaded_count += 1
                except pygame.error as e:
                    print(f"❌ 无法加载 {filepath}: {e}", file=sys.stderr)
            else:
                print(f"❌ 文件不存在: {filepath}", file=sys.stderr)
        
        if loaded_count == 0:
            print("❌ 没有加载到任何音频文件", file=sys.stderr)
            return False
        
        print(f"✅ 成功加载 {loaded_count}/5 个音频文件", file=sys.stderr)
        return True
    
    def play_finger_sound(self, finger_name, volume=0.7):
        """播放指定手指的音频"""
        if finger_name in self.sounds:
            try:
                sound = self.sounds[finger_name]
                sound.set_volume(volume)
                channel = pygame.mixer.find_channel()
                
                if channel:
                    channel.play(sound)
                    print(f"🎵 播放: {finger_name} (音量: {volume:.2f})", file=sys.stderr)
                    return True
                else:
                    print(f"⚠️ 没有可用通道播放 {finger_name}", file=sys.stderr)
                    
            except pygame.error as e:
                print(f"❌ 播放失败 {finger_name}: {e}", file=sys.stderr)
        
        return False
    
    def process_gesture_data(self, gesture_data):
        """处理手势数据并播放相应音频"""
        current_time = time.time() * 1000  # 毫秒
        
        try:
            # 解析JSON数据
            if isinstance(gesture_data, str):
                data = json.loads(gesture_data.strip())
            else:
                data = gesture_data
            
            # 获取手指状态和归一化角度
            finger_states = data.get("states", {})
            normalized_angles = data.get("normalized_angles", {})
            
            # 处理每个手指
            for finger_name in self.finger_states.keys():
                if finger_name in finger_states:
                    is_bent = finger_states[finger_name]
                    normalized_angle = normalized_angles.get(finger_name, 0.0)
                    
                    # 获取当前手指状态
                    finger_info = self.finger_states[finger_name]
                    last_bent = finger_info["bent"]
                    last_change_time = finger_info["last_change"]
                    
                    # 检查状态变化和防抖
                    state_changed = (is_bent != last_bent)
                    time_since_change = current_time - last_change_time
                    debounce_ok = time_since_change > self.debounce_time
                    
                    # 如果从伸直变为弯曲，且通过防抖检查
                    if state_changed and is_bent and debounce_ok:
                        # 根据弯曲程度调整音量 (归一化角度越小，弯曲越多，音量越大)
                        volume = max(0.3, min(1.0, 1.0 - normalized_angle))
                        
                        if self.play_finger_sound(finger_name, volume):
                            finger_info["bent"] = is_bent
                            finger_info["last_change"] = current_time
                    
                    # 更新状态（即使不播放音频）
                    elif state_changed and debounce_ok:
                        finger_info["bent"] = is_bent
                        finger_info["last_change"] = current_time
                        
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ 处理手势数据出错: {e}", file=sys.stderr)
    
    def run(self):
        """运行音频播放器，从标准输入读取数据"""
        print("🎧 音频播放器已启动", file=sys.stderr)
        print("💡 等待手势数据...", file=sys.stderr)
        print("🎹 大拇指=do, 食指=re, 中指=mi, 无名指=sol, 小指=la", file=sys.stderr)
        print("-" * 40, file=sys.stderr)
        
        try:
            while True:
                try:
                    # 从标准输入读取一行
                    line = sys.stdin.readline()
                    
                    if not line:  # EOF
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
        try:
            pygame.mixer.quit()
            print("🎵 音频系统已关闭", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}", file=sys.stderr)

def main():
    """主函数"""
    # 检查sounds文件夹
    if not os.path.exists("sounds"):
        print("❌ sounds文件夹不存在", file=sys.stderr)
        print("💡 请确保 five_tones.py 已运行并生成了音频文件", file=sys.stderr)
        sys.exit(1)
    
    # 创建并运行音频播放器
    try:
        player = GestureAudioPlayer()
        player.run()
    except Exception as e:
        print(f"❌ 播放器启动失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()