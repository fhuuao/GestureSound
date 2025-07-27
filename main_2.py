import cv2
import mediapipe as mp # type: ignore
import time
import threading
import json
import math
import os
import sys
from collections import deque
import traceback
import gc
import queue
import pygame
import numpy as np

# 导入自动串口连接模块
try:
    from auto_mcu_comm import MicrocontrollerConnection
except ImportError:
    print("❌ 无法导入 MicrocontrollerConnection，请确保 auto_mcu_comm.py 文件存在")
    print("⚠️ 将以纯音频模式运行")
    MicrocontrollerConnection = None

# 定义每个手指的角度范围
FINGER_ANGLE_RANGES = {
    "thumb": {"min": 120, "max": 180},
    "index": {"min": 5, "max": 180},
    "middle": {"min": 5, "max": 180},
    "ring": {"min": 5, "max": 180},
    "pinky": {"min": 5, "max": 180}
}

# 数据采集和发送配置
CAPTURE_FPS = 30               # 摄像头采集帧率
ARDUINO_AVERAGE_FRAMES = 5     # Arduino数据每N帧平均后发送一次
ARDUINO_SEND_FREQUENCY = CAPTURE_FPS / ARDUINO_AVERAGE_FRAMES  # 实际Arduino发送频率 6Hz

# 全局状态标志
system_running = True
system_error = False

class IntegratedAudioPlayer:
    """内置音频播放器"""
    
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
        self.debounce_time = 80  # 80ms防抖
        
        # 初始化音频
        if not self._init_audio():
            raise Exception("音频初始化失败")
        
        # 创建音频（短音调）
        self.sounds = {}
        for finger, freq in self.frequencies.items():
            self.sounds[finger] = self._create_tone(freq)
        
        print("✅ 内置音频播放器初始化完成")
    
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
                time.sleep(0.1)
                
                if config:
                    pygame.mixer.pre_init(**config)
                
                pygame.mixer.init()
                pygame.mixer.set_num_channels(8)
                
                # 测试基本功能
                freq, size, channels = pygame.mixer.get_init()
                print(f"✅ 音频系统初始化成功: {freq}Hz, {size}bit, {channels}ch")
                return True
                
            except Exception as e:
                print(f"⚠️ 音频配置{i+1}失败: {e}")
                continue
        
        print("❌ 所有音频配置都失败")
        return False
    
    def _create_tone(self, frequency, duration=0.3):
        """创建短音调"""
        try:
            # 获取当前音频设置
            freq, size, channels = pygame.mixer.get_init()
            sample_rate = freq
            samples = int(sample_rate * duration)
            
            # 生成正弦波
            t = np.linspace(0, duration, samples, False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.25  # 降低音量
            
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
            print(f"❌ 创建音调失败 {frequency}Hz: {e}")
            return None
    
    def play_finger(self, finger):
        """播放手指音调"""
        try:
            current_time = time.time() * 1000
            
            # 防抖检查
            if current_time - self.last_change[finger] < self.debounce_time:
                return
            
            if finger in self.sounds and self.sounds[finger] and not self.playing[finger]:
                # 停止之前的播放
                if finger in self.channels and self.channels[finger]:
                    try:
                        self.channels[finger].stop()
                    except:
                        pass
                
                # 播放新的音调
                channel = self.sounds[finger].play()
                if channel:
                    self.channels[finger] = channel
                    self.playing[finger] = True
                    self.last_change[finger] = current_time
                    
        except Exception as e:
            print(f"❌ 播放失败 {finger}: {e}")
    
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
                
        except Exception as e:
            print(f"❌ 停止失败 {finger}: {e}")
    
    def update_finger_states(self, states_data):
        """更新手指状态并播放音频"""
        try:
            for finger in self.frequencies:
                if finger in states_data:
                    if states_data[finger]:  # 弯曲
                        if not self.playing[finger]:  # 只有在没有播放时才开始播放
                            self.play_finger(finger)
                    else:  # 伸直
                        if self.playing[finger]:  # 只有在播放时才停止
                            self.stop_finger(finger)
        except Exception as e:
            print(f"❌ 更新手指状态错误: {e}")
    
    def cleanup_dead_channels(self):
        """清理已结束的通道"""
        try:
            for finger in list(self.channels.keys()):
                if finger in self.channels and self.channels[finger]:
                    if not self.channels[finger].get_busy():
                        self.playing[finger] = False
                        del self.channels[finger]
        except Exception as e:
            print(f"⚠️ 清理通道时出错: {e}")
    
    def stop_all(self):
        """停止所有播放"""
        for finger in self.frequencies:
            if self.playing[finger]:
                try:
                    if finger in self.channels and self.channels[finger]:
                        self.channels[finger].stop()
                    self.playing[finger] = False
                except Exception as e:
                    print(f"⚠️ 停止{finger}时出错: {e}")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理内置音频播放器...")
        self.stop_all()
        
        try:
            time.sleep(0.1)
            pygame.mixer.quit()
            print("🎵 音频系统已关闭")
        except Exception as e:
            print(f"⚠️ 清理时出错: {e}")

def normalize_angle(angle, finger_name):
    """将角度归一化到0-1范围"""
    if finger_name not in FINGER_ANGLE_RANGES:
        return 0.0
    
    min_angle = FINGER_ANGLE_RANGES[finger_name]["min"]
    max_angle = FINGER_ANGLE_RANGES[finger_name]["max"]
    
    clamped_angle = max(min_angle, min(max_angle, angle))
    normalized = (clamped_angle - min_angle) / (max_angle - min_angle)
    
    return round(normalized, 3)

def normalize_angles_dict(angles_dict):
    """批量归一化角度字典"""
    normalized_angles = {}
    for finger, angle in angles_dict.items():
        normalized_angles[finger] = normalize_angle(angle, finger)
    return normalized_angles

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.6, trackCon=0.3):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        try:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
            return frame
        except Exception as e:
            print(f"❌ HandDetector.findHands 错误: {e}")
            return frame
    
    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []
        try:
            if hasattr(self, 'results') and self.results.multi_hand_landmarks:
                if handNo < len(self.results.multi_hand_landmarks):
                    myHand = self.results.multi_hand_landmarks[handNo]

                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                        if draw and id == 0:
                            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), -1)
        except Exception as e:
            print(f"❌ HandDetector.findPosition 错误: {e}")
        return lmList

def calculate_angle(point1, point2, point3):
    """计算三个点之间的角度"""
    try:
        v1 = [point1[0] - point2[0], point1[1] - point2[1]]
        v2 = [point3[0] - point2[0], point3[1] - point2[1]]
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        v1_length = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_length = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if v1_length == 0 or v2_length == 0:
            return 0
        
        cos_angle = dot_product / (v1_length * v2_length)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    except Exception as e:
        print(f"❌ calculate_angle 错误: {e}")
        return 0

def calculate_distance(point1, point2):
    """计算两点间距离"""
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except Exception as e:
        print(f"❌ calculate_distance 错误: {e}")
        return 0

def is_hand_left_or_right(lmList):
    """判断是左手还是右手"""
    try:
        if len(lmList) < 21:
            return "unknown"
        
        wrist = lmList[0]
        middle_mcp = lmList[9]
        thumb_tip = lmList[4]
        
        if thumb_tip[1] < middle_mcp[1]:
            return "right"
        else:
            return "left"
    except Exception as e:
        print(f"❌ is_hand_left_or_right 错误: {e}")
        return "unknown"

def calculate_thumb_improved(lmList):
    """改进的拇指检测算法"""
    try:
        if len(lmList) < 5:
            return 0, False
        
        wrist = [lmList[0][1], lmList[0][2]]
        thumb_cmc = [lmList[1][1], lmList[1][2]]
        thumb_mcp = [lmList[2][1], lmList[2][2]]
        thumb_ip = [lmList[3][1], lmList[3][2]]
        thumb_tip = [lmList[4][1], lmList[4][2]]
        
        index_mcp = [lmList[5][1], lmList[5][2]]
        middle_mcp = [lmList[9][1], lmList[9][2]]
        
        angle_cmc_mcp_ip = calculate_angle(thumb_cmc, thumb_mcp, thumb_ip)
        angle_mcp_ip_tip = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        
        tip_to_wrist = calculate_distance(thumb_tip, wrist)
        tip_to_index = calculate_distance(thumb_tip, index_mcp)
        mcp_to_wrist = calculate_distance(thumb_mcp, wrist)
        
        palm_vector = [middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]]
        thumb_vector = [thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1]]
        
        palm_length = math.sqrt(palm_vector[0]**2 + palm_vector[1]**2)
        if palm_length > 0:
            projection = (thumb_vector[0] * palm_vector[0] + thumb_vector[1] * palm_vector[1]) / palm_length
            projection_ratio = projection / palm_length
        else:
            projection_ratio = 0
        
        hand_type = is_hand_left_or_right(lmList)
        lateral_offset = thumb_tip[0] - thumb_mcp[0]
        
        if hand_type == "right":
            lateral_bent = lateral_offset > -20
        else:
            lateral_bent = lateral_offset < 20
        
        detection_scores = []
        
        angle_score = 0
        if angle_cmc_mcp_ip < 160:
            angle_score += 0.5
        if angle_mcp_ip_tip < 160:
            angle_score += 0.5
        detection_scores.append(angle_score)
        
        distance_score = 0
        if mcp_to_wrist > 0:
            tip_wrist_ratio = tip_to_wrist / mcp_to_wrist
            if tip_wrist_ratio < 1.3:
                distance_score += 0.5
        
        if tip_to_index < 60:
            distance_score += 0.5
        detection_scores.append(distance_score)
        
        projection_score = 1.0 if projection_ratio < 0.5 else 0.0
        detection_scores.append(projection_score)
        
        lateral_score = 1.0 if lateral_bent else 0.0
        detection_scores.append(lateral_score)
        
        weights = [0.3, 0.25, 0.25, 0.2]
        final_score = sum(score * weight for score, weight in zip(detection_scores, weights))
        
        is_bent = final_score > 0.4
        main_angle = (angle_cmc_mcp_ip + angle_mcp_ip_tip) / 2
        
        return main_angle, is_bent
    except Exception as e:
        print(f"❌ calculate_thumb_improved 错误: {e}")
        return 0, False

def calculate_finger_angles_and_states(lmList):
    """计算手指角度并智能判断弯曲状态"""
    angles = {
        "thumb": 0,
        "index": 0,
        "middle": 0,
        "ring": 0,
        "pinky": 0
    }
    
    states = {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }
    
    try:
        if len(lmList) < 21:
            return angles, states
        
        # 拇指特殊处理
        if len(lmList) >= 5:
            angles["thumb"], states["thumb"] = calculate_thumb_improved(lmList)
        
        # 食指
        if len(lmList) >= 9:
            angles["index"] = calculate_angle(
                [lmList[5][1], lmList[5][2]],
                [lmList[6][1], lmList[6][2]],
                [lmList[8][1], lmList[8][2]]
            )
            states["index"] = angles["index"] < 50
        
        # 中指
        if len(lmList) >= 13:
            angles["middle"] = calculate_angle(
                [lmList[9][1], lmList[9][2]],
                [lmList[10][1], lmList[10][2]],
                [lmList[12][1], lmList[12][2]]
            )
            states["middle"] = angles["middle"] < 50
        
        # 无名指
        if len(lmList) >= 17:
            angles["ring"] = calculate_angle(
                [lmList[13][1], lmList[13][2]],
                [lmList[14][1], lmList[14][2]],
                [lmList[16][1], lmList[16][2]]
            )
            states["ring"] = angles["ring"] < 60
        
        # 小指
        if len(lmList) >= 21:
            angles["pinky"] = calculate_angle(
                [lmList[17][1], lmList[17][2]],
                [lmList[18][1], lmList[18][2]],
                [lmList[20][1], lmList[20][2]]
            )
            states["pinky"] = angles["pinky"] < 120
    
    except Exception as e:
        print(f"❌ calculate_finger_angles_and_states 错误: {e}")
    
    return angles, states

def send_to_arduino(mcu_connection, normalized_angles):
    """发送归一化角度数据到Arduino"""
    try:
        if mcu_connection and mcu_connection.connection and mcu_connection.connection.is_open:
            json_msg = json.dumps(normalized_angles, separators=(',', ':')) + '\n'
            mcu_connection.send(json_msg)
            return True
    except Exception as e:
        print(f"❌ Arduino串口发送错误: {e}")
        return False
    return False

def setup_mcu_connection():
    """设置单片机连接"""
    print("\n📡 开始自动检测并连接单片机...")
    try:
        mcu = MicrocontrollerConnection(baudrate=9600, timeout=1)
        if mcu.auto_connect():
            print(f"✅ 已成功连接到单片机: {mcu.connection.port}")
            return mcu
        else:
            print("⚠️ 无法自动连接到单片机，将继续运行但不发送数据到Arduino")
            return None
    except Exception as e:
        print(f"⚠️ 串口连接设置失败: {e} (将继续运行，但不发送数据到Arduino)")
        return None

def arduino_sender_thread(mcu_connection, arduino_data_queue):
    """Arduino数据发送线程"""
    global system_running, system_error
    
    print(f"📡 Arduino发送线程启动 (频率: {ARDUINO_SEND_FREQUENCY:.1f}Hz)")
    
    arduino_count = 0
    error_count = 0
    last_heartbeat = time.time()
    
    try:
        while system_running and not system_error:
            try:
                current_time = time.time()
                
                # 心跳检测
                if current_time - last_heartbeat > 20:
                    arduino_queue_size = arduino_data_queue.qsize()
                    print(f"💗 Arduino线程心跳: 发送={arduino_count}, 错误={error_count}, 队列={arduino_queue_size}")
                    last_heartbeat = current_time
                    
                    if error_count > 50:
                        print("❌ Arduino发送错误过多")
                        break
                    error_count = 0
                
                # 发送Arduino数据
                try:
                    arduino_data = arduino_data_queue.get(timeout=0.1)
                    arduino_data_queue.task_done()
                    
                    if mcu_connection and send_to_arduino(mcu_connection, arduino_data):
                        arduino_count += 1
                    elif mcu_connection:
                        error_count += 1
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    error_count += 1
                
            except Exception as e:
                print(f"❌ Arduino线程内部错误: {e}")
                error_count += 1
                time.sleep(0.1)
                
    except Exception as e:
        print(f"❌ Arduino发送线程严重错误: {e}")
        print(traceback.format_exc())
    
    print(f"📡 Arduino线程退出: 发送={arduino_count}")

def main():
    global system_running, system_error
    
    print("🎵 内置音频版手势识别系统")
    print("=" * 50)
    print(f"📊 配置信息:")
    print(f"   摄像头采集: {CAPTURE_FPS}FPS")
    print(f"   音频播放: 内置直接调用")
    print(f"   Arduino发送: {ARDUINO_SEND_FREQUENCY:.1f}Hz (每{ARDUINO_AVERAGE_FRAMES}帧平均)")
    
    # 初始化内置音频播放器
    audio_player = None
    try:
        audio_player = IntegratedAudioPlayer()
    except Exception as e:
        print(f"❌ 音频播放器初始化失败: {e}")
        return
    
    # 设置单片机连接
    mcu_connection = None
    if MicrocontrollerConnection:
        mcu_connection = setup_mcu_connection()
    
    # 创建Arduino数据队列
    arduino_data_queue = queue.Queue(maxsize=2)
    
    # 启动Arduino发送线程
    arduino_thread = None
    if mcu_connection:
        arduino_thread = threading.Thread(
            target=arduino_sender_thread, 
            args=(mcu_connection, arduino_data_queue), 
            daemon=True
        )
        arduino_thread.start()
    
    # 主循环变量
    prevTime = 0
    frame_count = 0
    last_gc_time = time.time()
    last_status_time = time.time()
    last_cleanup_time = time.time()
    
    # Arduino数据平均缓存
    arduino_angle_buffer = {
        "thumb": deque(maxlen=ARDUINO_AVERAGE_FRAMES),
        "index": deque(maxlen=ARDUINO_AVERAGE_FRAMES),
        "middle": deque(maxlen=ARDUINO_AVERAGE_FRAMES),
        "ring": deque(maxlen=ARDUINO_AVERAGE_FRAMES),
        "pinky": deque(maxlen=ARDUINO_AVERAGE_FRAMES)
    }

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        detector = HandDetector(maxHands=1, detectionCon=0.6, trackCon=0.3)
        
        cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Control", 640, 500)
        
        print("\n🚀 系统启动完成！")
        print("📸 摄像头已就绪")
        print("🎵 内置音频播放器已就绪")
        if mcu_connection:
            print("📡 Arduino串口通信已就绪")
        print("💡 弯曲手指即可播放音乐！")
        print("❌ 按 'q' 退出程序")
        print("-" * 50)

        arduino_send_count = 0

        while system_running and not system_error:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                    
                frame_count += 1
                current_time = time.time()
                
                # 定期垃圾回收
                if current_time - last_gc_time > 60:
                    gc.collect()
                    last_gc_time = current_time
                
                # 定期清理音频通道
                if current_time - last_cleanup_time > 2:
                    audio_player.cleanup_dead_channels()
                    last_cleanup_time = current_time
                
                # 定期状态检查
                if current_time - last_status_time > 30:
                    arduino_queue_size = arduino_data_queue.qsize() if mcu_connection else 0
                    playing_count = sum(1 for p in audio_player.playing.values() if p)
                    print(f"💗 主线程心跳: Frame={frame_count}, 音频播放={playing_count}/5, "
                          f"ArduinoQ={arduino_queue_size}, ArduinoSent={arduino_send_count}")
                    
                    last_status_time = current_time
                    arduino_send_count = 0
                
                # 手势识别
                frame = detector.findHands(frame)
                lmList = detector.findPosition(frame)
                
                if lmList:
                    # 检测到手部
                    current_angles, current_states = calculate_finger_angles_and_states(lmList)
                else:
                    # 检测不到手时的默认值
                    current_angles = {"thumb": 180, "index": 180, "middle": 180, "ring": 180, "pinky": 180}
                    current_states = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
                
                # 直接更新音频播放器状态（零延迟）
                audio_player.update_finger_states(current_states)
                
                # Arduino数据累积和平均处理
                normalized_angles = normalize_angles_dict(current_angles)
                
                # 添加到Arduino缓存
                for finger in arduino_angle_buffer:
                    arduino_angle_buffer[finger].append(normalized_angles[finger])
                
                # 每N帧计算平均值并发送Arduino数据
                if frame_count % ARDUINO_AVERAGE_FRAMES == 0 and mcu_connection:
                    # 计算平均值
                    averaged_angles = {}
                    for finger in arduino_angle_buffer:
                        if len(arduino_angle_buffer[finger]) > 0:
                            averaged_angles[finger] = round(
                                sum(arduino_angle_buffer[finger]) / len(arduino_angle_buffer[finger]), 3
                            )
                        else:
                            averaged_angles[finger] = 1.0  # 默认值（伸直）
                    
                    # 发送平均后的Arduino数据
                    try:
                        arduino_data_queue.put_nowait(averaged_angles)
                        arduino_send_count += 1
                    except queue.Full:
                        try:
                            arduino_data_queue.get_nowait()
                            arduino_data_queue.task_done()
                            arduino_data_queue.put_nowait(averaged_angles)
                            arduino_send_count += 1
                        except:
                            pass
                
                # 计算FPS
                if prevTime != 0:
                    fps = 1 / (current_time - prevTime)
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                prevTime = current_time
                
                # 显示角度信息
                y_offset = 60
                fingers = ["thumb", "index", "middle", "ring", "pinky"]
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
                if lmList:
                    for i, (finger, name) in enumerate(zip(fingers, finger_names)):
                        angle = current_angles[finger]
                        normalized = normalized_angles[finger]
                        state = current_states[finger]
                        playing = audio_player.playing[finger]
                        
                        # 显示颜色：绿色=伸直，红色=弯曲，蓝色=播放中
                        if playing:
                            color = (255, 0, 0)  # 蓝色 - 播放中
                        elif state:
                            color = (0, 0, 255)  # 红色 - 弯曲
                        else:
                            color = (0, 255, 0)  # 绿色 - 伸直
                        
                        state_text = "Playing" if playing else ("Bent" if state else "Straight")
                        text = f"{name}: {angle:.1f}° (N:{normalized:.3f}) {state_text}"
                        
                        cv2.putText(frame, text, (10, y_offset + i * 22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                else:
                    cv2.putText(frame, "No Hand Detected - All Audio Stopped", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 显示系统状态
                status_y = y_offset + 5 * 22 + 20
                
                # 音频状态
                playing_count = sum(1 for p in audio_player.playing.values() if p)
                audio_status = f"Audio: Built-in - {playing_count}/5 Playing"
                audio_color = (0, 255, 0) if playing_count > 0 else (0, 255, 255)
                cv2.putText(frame, audio_status, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, audio_color, 1)
                
                # Arduino状态
                if mcu_connection and mcu_connection.connection and mcu_connection.connection.is_open:
                    arduino_status = f"Arduino: {ARDUINO_SEND_FREQUENCY:.1f}Hz - Connected"
                    arduino_color = (0, 255, 0)
                else:
                    arduino_status = f"Arduino: {ARDUINO_SEND_FREQUENCY:.1f}Hz - Disconnected"
                    arduino_color = (0, 0, 255)
                cv2.putText(frame, arduino_status, (10, status_y + 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, arduino_color, 1)
                
                # 队列状态
                arduino_queue_size = arduino_data_queue.qsize() if mcu_connection else 0
                cv2.putText(frame, f"ArduinoQ: {arduino_queue_size}/2, Direct Audio", 
                           (10, status_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                
                # 系统状态
                if system_error:
                    cv2.putText(frame, "SYSTEM ERROR!", (10, status_y + 54), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "System OK", (10, status_y + 54), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                cv2.imshow("Hand Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except Exception as e:
                print(f"❌ 主循环错误: {e}")
                time.sleep(0.1)

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print(traceback.format_exc())
        system_error = True
    
    finally:
        # 设置停止标志
        system_running = False
        
        print("🧹 开始清理资源...")
        
        # 清理摄像头
        if 'cap' in locals():
            try:
                cap.release()
                print("📸 摄像头已释放")
            except:
                pass
        
        # 清理OpenCV窗口
        try:
            cv2.destroyAllWindows()
            print("🖼️ OpenCV窗口已关闭")
        except:
            pass
        
        # 关闭串口连接
        if mcu_connection:
            try:
                mcu_connection.close()
                print("📡 Arduino连接已关闭")
            except Exception as e:
                print(f"⚠️ 关闭Arduino连接时出错: {e}")
        
        # 等待Arduino线程结束
        if arduino_thread:
            try:
                arduino_thread.join(timeout=3)
                print("📡 Arduino线程已结束")
            except:
                print("⚠️ Arduino线程强制结束")
        
        # 清理内置音频播放器
        if audio_player:
            try:
                audio_player.cleanup()
            except Exception as e:
                print(f"⚠️ 清理音频播放器时出错: {e}")
        
        # 清理队列
        try:
            while not arduino_data_queue.empty():
                arduino_data_queue.get_nowait()
                arduino_data_queue.task_done()
            print("📦 数据队列已清空")
        except:
            pass
        
        # 最终垃圾回收
        gc.collect()
        
        print("✅ 程序已完全退出")

if __name__ == "__main__":
    main()