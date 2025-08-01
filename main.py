import cv2
import mediapipe as mp # type: ignore
import time
import threading
import json
import math
import os
import sys
import subprocess
from collections import deque
import traceback
import gc
import queue

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
AUDIO_SEND_FREQUENCY = 30      # 音频数据发送频率 (Hz)
ARDUINO_AVERAGE_FRAMES = 5     # Arduino数据每N帧平均后发送一次
ARDUINO_SEND_FREQUENCY = CAPTURE_FPS / ARDUINO_AVERAGE_FRAMES  # 实际Arduino发送频率 6Hz

# 全局状态标志
system_running = True
system_error = False

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

def send_to_audio_player(audio_process, states_data):
    """发送手势状态数据到音频播放器"""
    try:
        if audio_process and audio_process.stdin and not audio_process.stdin.closed:
            json_msg = json.dumps(states_data, separators=(',', ':'))
            audio_process.stdin.write(json_msg + '\n')
            audio_process.stdin.flush()
            return True
    except Exception as e:
        return False
    return False

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

def data_sender_thread(audio_process, mcu_connection, audio_data_queue, arduino_data_queue):
    """优化的数据发送线程 - 分别处理音频和Arduino数据"""
    global system_running, system_error
    
    audio_interval = 1.0 / AUDIO_SEND_FREQUENCY
    last_audio_send = 0
    
    print(f"📡 数据发送线程启动:")
    print(f"   音频频率: {AUDIO_SEND_FREQUENCY}Hz")
    print(f"   Arduino频率: {ARDUINO_SEND_FREQUENCY:.1f}Hz (每{ARDUINO_AVERAGE_FRAMES}帧平均)")
    
    audio_count = 0
    arduino_count = 0
    error_count = 0
    last_heartbeat = time.time()
    
    try:
        while system_running and not system_error:
            try:
                current_time = time.time()
                
                # 心跳检测
                if current_time - last_heartbeat > 15:
                    audio_queue_size = audio_data_queue.qsize()
                    arduino_queue_size = arduino_data_queue.qsize()
                    print(f"💗 发送线程心跳: Audio={audio_count}, Arduino={arduino_count}, "
                          f"Errors={error_count}, AudioQ={audio_queue_size}, ArduinoQ={arduino_queue_size}")
                    last_heartbeat = current_time
                    
                    if error_count > 100:
                        print("❌ 发送线程错误过多，标记系统错误")
                        system_error = True
                        break
                    error_count = 0
                
                # 发送音频数据（高频）
                if current_time - last_audio_send >= audio_interval:
                    try:
                        audio_data = None
                        # 获取最新的音频数据
                        while not audio_data_queue.empty():
                            audio_data = audio_data_queue.get_nowait()
                            audio_data_queue.task_done()
                        
                        if audio_data and send_to_audio_player(audio_process, audio_data):
                            audio_count += 1
                        elif audio_data:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                    last_audio_send = current_time
                
                # 发送Arduino数据（低频，来自平均后的数据）
                try:
                    arduino_data = arduino_data_queue.get_nowait()
                    arduino_data_queue.task_done()
                    
                    if mcu_connection and send_to_arduino(mcu_connection, arduino_data):
                        arduino_count += 1
                    elif mcu_connection:
                        error_count += 1
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    error_count += 1
                
                time.sleep(0.005)  # 5ms间隔
                
            except Exception as e:
                print(f"❌ 发送线程内部错误: {e}")
                error_count += 1
                time.sleep(0.1)
                
    except Exception as e:
        print(f"❌ 数据发送线程严重错误: {e}")
        print(traceback.format_exc())
        system_error = True
    
    print(f"📡 发送线程退出: Audio={audio_count}, Arduino={arduino_count}")

def main():
    global system_running, system_error
    
    print("🎵 优化版手势识别音乐系统")
    print("=" * 50)
    print(f"📊 配置信息:")
    print(f"   摄像头采集: {CAPTURE_FPS}FPS")
    print(f"   音频发送: {AUDIO_SEND_FREQUENCY}Hz")
    print(f"   Arduino发送: {ARDUINO_SEND_FREQUENCY:.1f}Hz (每{ARDUINO_AVERAGE_FRAMES}帧平均)")
    
    # 启动音频播放器
    audio_process = None
    try:
        audio_process = subprocess.Popen(
            [sys.executable, "realtime_audio_player.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        time.sleep(2)
        
        if audio_process.poll() is None:
            print("✅ 实时音频播放器启动成功")
        else:
            print("❌ 实时音频播放器启动失败")
            return
            
    except FileNotFoundError:
        print("❌ 找不到 realtime_audio_player.py 文件")
        return
    except Exception as e:
        print(f"❌ 音频系统启动错误: {e}")
        return
    
    # 设置单片机连接
    mcu_connection = None
    if MicrocontrollerConnection:
        mcu_connection = setup_mcu_connection()
    
    # 创建两个独立的数据队列
    audio_data_queue = queue.Queue(maxsize=3)    # 音频数据队列
    arduino_data_queue = queue.Queue(maxsize=2)  # Arduino数据队列
    
    # 启动数据发送线程
    sender_thread = threading.Thread(
        target=data_sender_thread, 
        args=(audio_process, mcu_connection, audio_data_queue, arduino_data_queue), 
        daemon=True
    )
    sender_thread.start()
    
    # 主循环变量
    prevTime = 0
    frame_count = 0
    last_gc_time = time.time()
    last_status_time = time.time()
    
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
        print("🎵 音频播放已就绪")
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
                
                # 定期状态检查
                if current_time - last_status_time > 30:
                    audio_queue_size = audio_data_queue.qsize()
                    arduino_queue_size = arduino_data_queue.qsize()
                    print(f"💗 主线程心跳: Frame={frame_count}, AudioQ={audio_queue_size}, "
                          f"ArduinoQ={arduino_queue_size}, ArduinoSent={arduino_send_count}")
                    
                    # 检查音频进程状态
                    if audio_process and audio_process.poll() is not None:
                        print("⚠️ 音频进程已退出，标记系统错误")
                        system_error = True
                        break
                    
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
                
                # 每帧都发送音频数据（高频）
                try:
                    audio_data_queue.put_nowait(current_states.copy())
                except queue.Full:
                    try:
                        audio_data_queue.get_nowait()
                        audio_data_queue.task_done()
                        audio_data_queue.put_nowait(current_states.copy())
                    except:
                        pass
                
                # Arduino数据累积和平均处理
                normalized_angles = normalize_angles_dict(current_angles)
                
                # 添加到Arduino缓存
                for finger in arduino_angle_buffer:
                    arduino_angle_buffer[finger].append(normalized_angles[finger])
                
                # 每N帧计算平均值并发送Arduino数据
                if frame_count % ARDUINO_AVERAGE_FRAMES == 0:
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
                        color = (0, 255, 0) if not state else (0, 0, 255)
                        
                        state_text = "Bent" if state else "Straight"
                        text = f"{name}: {angle:.1f}° (N:{normalized:.3f}) {state_text}"
                        
                        cv2.putText(frame, text, (10, y_offset + i * 22), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                else:
                    cv2.putText(frame, "No Hand Detected - All Audio Stopped", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 显示系统状态
                status_y = y_offset + 5 * 22 + 20
                
                # 音频状态
                if audio_process and audio_process.poll() is None:
                    audio_status = f"Audio: {AUDIO_SEND_FREQUENCY}Hz - Running"
                    audio_color = (0, 255, 0)
                else:
                    audio_status = f"Audio: {AUDIO_SEND_FREQUENCY}Hz - Stopped"
                    audio_color = (0, 0, 255)
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
                audio_queue_size = audio_data_queue.qsize()
                arduino_queue_size = arduino_data_queue.qsize()
                cv2.putText(frame, f"AudioQ: {audio_queue_size}/3, ArduinoQ: {arduino_queue_size}/2", 
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
        
        # 等待发送线程结束
        if 'sender_thread' in locals():
            try:
                sender_thread.join(timeout=3)
                print("📡 发送线程已结束")
            except:
                print("⚠️ 发送线程强制结束")
        
        # 关闭音频播放器进程
        if audio_process:
            try:
                if audio_process.stdin and not audio_process.stdin.closed:
                    audio_process.stdin.close()
                audio_process.terminate()
                
                # 等待进程结束
                try:
                    audio_process.wait(timeout=5)
                    print("🎵 音频播放器已正常关闭")
                except subprocess.TimeoutExpired:
                    audio_process.kill()
                    audio_process.wait()
                    print("🎵 音频播放器已强制关闭")
            except Exception as e:
                print(f"⚠️ 关闭音频播放器时出错: {e}")
        
        # 清理队列
        try:
            while not audio_data_queue.empty():
                audio_data_queue.get_nowait()
                audio_data_queue.task_done()
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