# 空气钢琴
import cv2
import mediapipe as mp
import time
import serial
import threading
from collections import deque
import pygame
import os

class HandDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.5):
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
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []

        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmList.append([id, cx, cy])

                    if draw and id == 0:
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), -1)
        return lmList

class AudioManager:
    """音频管理类"""
    def __init__(self, audio_folder="sounds"):
        """
        初始化音频管理器
        audio_folder: 音频文件夹路径，默认为"sounds"
        """
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.audio_folder = audio_folder
        
        # 定义音频文件映射：大拇指-do, 食指-re, 中指-mi, 无名指-sol, 小指-la
        self.audio_files = {
            4: "thumb.wav",     # 大拇指 - do
            1: "index.wav",     # 食指 - re
            2: "middle.wav",    # 中指 - mi
            3: "ring.wav",      # 无名指 - sol
            5: "pinky.wav"      # 小指 - la
        }
        
        # 音调说明
        self.note_mapping = {
            4: "do", 1: "re", 2: "mi", 3: "sol", 5: "la"
        }
        
        # 预加载音频文件
        self.sounds = {}
        self.load_audio_files()
        
        # 记录上次播放时间，避免频繁播放
        self.last_play_time = {i: 0 for i in range(6)}
        self.min_interval = 0.5  # 最小播放间隔（秒）
    
    def load_audio_files(self):
        """加载音频文件"""
        for finger_id, filename in self.audio_files.items():
            file_path = os.path.join(self.audio_folder, filename)
            try:
                if os.path.exists(file_path):
                    self.sounds[finger_id] = pygame.mixer.Sound(file_path)
                    print(f"[Audio] Loaded: {filename}")
                else:
                    print(f"[Audio] Warning: {file_path} not found")
                    # 创建一个简单的提示音作为替代
                    self.create_beep_sound(finger_id)
            except Exception as e:
                print(f"[Audio] Error loading {filename}: {e}")
                self.create_beep_sound(finger_id)
    
    def create_beep_sound(self, finger_id):
        """创建简单的提示音"""
        try:
            # 创建不同音调的提示音
            frequencies = {1: 440, 2: 523, 3: 659, 4: 784, 5: 880}  # A4, C5, E5, G5, A5
            freq = frequencies.get(finger_id, 440)
            
            import numpy as np
            duration = 0.2  # 秒
            sample_rate = 22050
            frames = int(duration * sample_rate)
            
            # 生成正弦波
            arr = np.zeros((frames, 2))
            for i in range(frames):
                wave = 4096 * np.sin(2 * np.pi * freq * i / sample_rate)
                # 添加淡入淡出效果
                envelope = min(i / (frames * 0.1), 1, (frames - i) / (frames * 0.1))
                arr[i] = [wave * envelope, wave * envelope]
            
            arr = arr.astype(np.int16)
            self.sounds[finger_id] = pygame.mixer.sndarray.make_sound(arr)
            print(f"[Audio] Created beep for finger {finger_id}")
        except Exception as e:
            print(f"[Audio] Failed to create beep for finger {finger_id}: {e}")
    
    def play_audio(self, finger_id):
        """播放指定手指的音频"""
        current_time = time.time()
        
        # 检查是否需要限制播放频率
        if current_time - self.last_play_time[finger_id] < self.min_interval:
            return
        
        if finger_id in self.sounds:
            try:
                self.sounds[finger_id].play()
                self.last_play_time[finger_id] = current_time
                
                # 获取手指名称和音调用于显示
                finger_names = {1: "食指", 2: "中指", 3: "无名指", 4: "大拇指", 5: "小指"}
                finger_name = finger_names.get(finger_id, f"手指{finger_id}")
                note_name = self.note_mapping.get(finger_id, "?")
                print(f"[Audio] Playing {note_name} for {finger_name}")
            except Exception as e:
                print(f"[Audio] Error playing sound for finger {finger_id}: {e}")

def serial_monitor(ser):
    """独立线程监听Arduino串口输出"""
    while True:
        try:
            if ser.in_waiting > 0:
                arduino_data = ser.readline().decode('utf-8').strip()
                if arduino_data:
                    print(f"[Arduino]: {arduino_data}")
        except:
            break
        time.sleep(0.01)

def main():
    prevTime = 0
    hand = [["Wrist", False], ["IndexFinger", False], ["Middle", False], 
            ["Ring", False], ["Thumb", False], ["Pinky", False]]
    
    # 初始化音频管理器
    try:
        audio_manager = AudioManager()
        print("[Audio] Audio system initialized")
    except Exception as e:
        print(f"[Audio] Failed to initialize audio system: {e}")
        audio_manager = None
    
    # 帧计数器和处理间隔
    frame_count = 0
    PROCESSING_INTERVAL = 1  # 每帧都检测，但每5帧取平均值
    
    # 创建滑动窗口存储最近5帧的手指状态
    WINDOW_SIZE = 5
    finger_history = {
        0: deque(maxlen=WINDOW_SIZE),  # 手腕
        1: deque(maxlen=WINDOW_SIZE),  # 食指
        2: deque(maxlen=WINDOW_SIZE),  # 中指
        3: deque(maxlen=WINDOW_SIZE),  # 无名指
        4: deque(maxlen=WINDOW_SIZE),  # 拇指
        5: deque(maxlen=WINDOW_SIZE)   # 小指
    }
    
    # 初始化窗口数据
    for i in range(WINDOW_SIZE):
        for finger in finger_history:
            finger_history[finger].append(False)

    try:
        # 串口配置
        ser = serial.Serial(
            port="COM24",#根据实际情况调整
            baudrate=9600,
            timeout=0.1,  # 短超时，用于非阻塞读取q
            write_timeout=1
        )
        print(f"Serial port {ser.port} opened successfully")
        time.sleep(2)
        
        # 启动串口监听线程
        serial_thread = threading.Thread(target=serial_monitor, args=(ser,), daemon=True)
        serial_thread.start()
        print("Serial monitor started")
        
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Failed to open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detector = HandDetector(maxHands=1, detectionCon=0.7)
        
        print(f"System ready. Averaging over {WINDOW_SIZE} frames.")
        print("Press 'q' to quit.")
        print("Arduino output will be displayed with [Arduino] prefix")
        print("Audio mapping: 大拇指-do, 食指-re, 中指-mi, 无名指-sol, 小指-la")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            frame_count += 1
            
            # 始终检测手部并绘制关键点
            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame)
            
            # 每帧都检测手指状态，但只在必要时更新平均值
            current_state = [False] * 6  # 初始化当前帧的手指状态
            
            if len(lmList) > 0: 
                j = 1
                
                for i in range(1, 6):
                    if i == 1:  # 拇指检测
                        if lmList[4][1] > lmList[3][1]:
                            current_state[4] = True  # 拇指弯曲
                    else:  # 其他四指检测
                        finger_tip = i * 4
                        finger_pip = i * 4 - 2
                        
                        if finger_tip < len(lmList) and finger_pip < len(lmList):
                            if lmList[finger_tip][2] > lmList[finger_pip][2]:
                                current_state[j] = True  # 手指弯曲
                        
                        if j == 3:
                            j += 2
                        else:
                            j += 1
            
            # 更新滑动窗口数据
            for i in range(6):
                finger_history[i].append(current_state[i])
            
            # 每5帧计算一次平均值并决定最终状态
            if frame_count % WINDOW_SIZE == 0:
                change = False
                threshold = WINDOW_SIZE // 2  # 超过半数帧为True则认为弯曲
                
                for i in range(6):
                    # 计算平均值
                    count_true = sum(finger_history[i])
                    new_state = count_true > threshold
                    
                    # 如果状态变化，记录变化
                    if new_state != hand[i][1]:
                        old_state = hand[i][1]
                        hand[i][1] = new_state
                        change = True
                        print(f"[Python] Frame {frame_count}: {hand[i][0]}: {'BENT' if new_state else 'STRAIGHT'}")
                        
                        # 新增：如果手指从直立变为弯曲，播放对应音频
                        if not old_state and new_state and audio_manager:
                            # 只有当手指从直立变为弯曲时才播放音频
                            if i in [1, 2, 3, 4, 5]:  # 排除手腕（索引0）
                                audio_manager.play_audio(i)
                
                # 如果状态变化，发送新命令
                if change:
                    msg = ""
                    for i in range(6):
                        if hand[i][1]:
                            msg += "1"
                        else:
                            msg += "0"

                    msg += '\n'
                    print(f"[Python] Sending: {msg.strip()}")
                    
                    try:
                        ser.write(msg.encode("ascii"))
                        ser.flush()
                    except serial.SerialException as e:
                        print(f"[Python] Serial write error: {e}")
            
            # 计算并显示实际FPS
            currentTime = time.time()
            if prevTime != 0:
                fps = 1 / (currentTime - prevTime)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            prevTime = currentTime
            
            # 显示处理参数
            cv2.putText(frame, f"Avg frames: {WINDOW_SIZE}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 添加状态显示
            y_offset = 90
            for i, (name, state) in enumerate(hand):
                color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(frame, f"{name}: {'BENT' if state else 'STRAIGHT'}", 
                           (10, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)

            # 添加音频状态显示
            if audio_manager:
                cv2.putText(frame, "Audio: ON", (10, y_offset + 6 * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Audio: OFF", (10, y_offset + 6 * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
        if audio_manager:
            pygame.mixer.quit()
            print("Audio system closed")
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()