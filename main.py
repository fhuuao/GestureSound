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

# 导入自动串口连接模块
try:
    from auto_mcu_comm import MicrocontrollerConnection
except ImportError:
    print("❌ 无法导入 MicrocontrollerConnection，请确保 auto_mcu_comm.py 文件存在")
    sys.exit(1)

# 定义每个手指的角度范围（根据实际测试调整）
FINGER_ANGLE_RANGES = {
    "thumb": {"min": 120, "max": 180},   # 拇指角度范围：120°=弯曲(0), 180°=伸直(1)
    "index": {"min": 5, "max": 180},     # 食指角度范围
    "middle": {"min": 5, "max": 180},    # 中指角度范围
    "ring": {"min": 5, "max": 180},      # 无名指角度范围
    "pinky": {"min": 5, "max": 180}      # 小指角度范围
}

# 数据发送频率配置
ARDUINO_SEND_FREQUENCY = 20    # Arduino数据发送频率 (Hz) - 20Hz适合舵机控制
AUDIO_SEND_FREQUENCY = 30      # 音频数据发送频率 (Hz) - 30Hz保证音频响应性

# 添加全局状态标志
system_running = True
system_error = False

def normalize_angle(angle, finger_name):
    """
    将角度归一化到0-1范围
    0 = 完全弯曲 (最小角度)
    1 = 完全伸直 (最大角度)
    """
    if finger_name not in FINGER_ANGLE_RANGES:
        return 0.0
    
    min_angle = FINGER_ANGLE_RANGES[finger_name]["min"]
    max_angle = FINGER_ANGLE_RANGES[finger_name]["max"]
    
    # 限制角度在定义范围内
    clamped_angle = max(min_angle, min(max_angle, angle))
    
    # 归一化到0-1范围
    normalized = (clamped_angle - min_angle) / (max_angle - min_angle)
    
    return round(normalized, 3)

def normalize_angles_dict(angles_dict):
    """
    批量归一化角度字典
    """
    normalized_angles = {}
    for finger, angle in angles_dict.items():
        normalized_angles[finger] = normalize_angle(angle, finger)
    return normalized_angles

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.4):
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
            if self.results.multi_hand_landmarks:
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
        # 计算向量
        v1 = [point1[0] - point2[0], point1[1] - point2[1]]
        v2 = [point3[0] - point2[0], point3[1] - point2[1]]
        
        # 计算点积和叉积
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # 计算向量长度
        v1_length = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_length = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if v1_length == 0 or v2_length == 0:
            return 0
        
        # 计算角度
        cos_angle = dot_product / (v1_length * v2_length)
        cos_angle = max(-1, min(1, cos_angle))  # 确保在[-1,1]范围内
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
        
        # 使用手腕(0)和中指MCP(9)的位置关系判断
        wrist = lmList[0]
        middle_mcp = lmList[9]
        thumb_tip = lmList[4]
        
        # 如果拇指在中指的左边，通常是右手；在右边通常是左手
        if thumb_tip[1] < middle_mcp[1]:  # 拇指在中指左边
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
        
        # 获取拇指关键点
        wrist = [lmList[0][1], lmList[0][2]]      # 手腕 (0)
        thumb_cmc = [lmList[1][1], lmList[1][2]]  # 拇指CMC关节 (1)
        thumb_mcp = [lmList[2][1], lmList[2][2]]  # 拇指MCP关节 (2)
        thumb_ip = [lmList[3][1], lmList[3][2]]   # 拇指IP关节 (3)
        thumb_tip = [lmList[4][1], lmList[4][2]]  # 拇指指尖 (4)
        
        # 获取其他参考点
        index_mcp = [lmList[5][1], lmList[5][2]]  # 食指MCP关节 (5)
        middle_mcp = [lmList[9][1], lmList[9][2]] # 中指MCP关节 (9)
        
        # 计算角度
        angle_cmc_mcp_ip = calculate_angle(thumb_cmc, thumb_mcp, thumb_ip)
        angle_mcp_ip_tip = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
        
        # 距离比较
        tip_to_wrist = calculate_distance(thumb_tip, wrist)
        tip_to_index = calculate_distance(thumb_tip, index_mcp)
        mcp_to_wrist = calculate_distance(thumb_mcp, wrist)
        
        # 位置关系判断
        palm_vector = [middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]]
        thumb_vector = [thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1]]
        
        palm_length = math.sqrt(palm_vector[0]**2 + palm_vector[1]**2)
        if palm_length > 0:
            projection = (thumb_vector[0] * palm_vector[0] + thumb_vector[1] * palm_vector[1]) / palm_length
            projection_ratio = projection / palm_length
        else:
            projection_ratio = 0
        
        # 左右手判断
        hand_type = is_hand_left_or_right(lmList)
        lateral_offset = thumb_tip[0] - thumb_mcp[0]
        
        if hand_type == "right":
            lateral_bent = lateral_offset > -20
        else:
            lateral_bent = lateral_offset < 20
        
        # 综合评分
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

def serial_monitor(mcu_connection):
    """独立线程监听Arduino串口输出"""
    global system_running
    try:
        while system_running:
            try:
                if mcu_connection and mcu_connection.connection and mcu_connection.connection.is_open:
                    arduino_data = mcu_connection.receive(timeout=0.1)
                    if arduino_data:
                        print(f"[Arduino]: {arduino_data.strip()}")
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"串口监听错误: {e}")
                time.sleep(0.5)  # 错误后等待更长时间
            time.sleep(0.01)
    except Exception as e:
        print(f"❌ serial_monitor 线程错误: {e}")
        print(traceback.format_exc())

def check_and_install_dependencies():
    """检查并提示安装必要的依赖"""
    missing_deps = []
    
    try:
        import pygame
    except ImportError:
        missing_deps.append("pygame")
    
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        missing_deps.append("numpy scipy")
    
    if missing_deps:
        print("❌ 缺少必要的依赖包，请安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def setup_audio_system():
    """设置音频系统：启动实时音频播放器"""
    print("🎵 设置实时音频系统...")
    
    print("🎧 启动实时音频播放器...")
    try:
        audio_process = subprocess.Popen(
            [sys.executable, "realtime_audio_player.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        time.sleep(1)
        
        if audio_process.poll() is None:
            print("✅ 实时音频播放器启动成功")
            return audio_process
        else:
            print("❌ 实时音频播放器启动失败")
            return None
            
    except FileNotFoundError:
        print("❌ 找不到 realtime_audio_player.py 文件")
        return None
    except Exception as e:
        print(f"❌ 音频系统启动错误: {e}")
        return None

def send_to_audio_player(audio_process, states_data):
    """发送手势状态数据到音频播放器"""
    try:
        if audio_process and audio_process.stdin and not audio_process.stdin.closed:
            json_msg = json.dumps(states_data, separators=(',', ':'))
            audio_process.stdin.write(json_msg + '\n')
            audio_process.stdin.flush()
            return True
    except (BrokenPipeError, OSError, ValueError) as e:
        print(f"⚠️ 音频播放器连接断开: {e}")
        return False
    except Exception as e:
        print(f"❌ 发送音频数据错误: {e}")
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

def data_sender_thread(audio_process, mcu_connection, gesture_data_queue):
    """数据发送线程 - 以固定频率发送数据"""
    global system_running, system_error
    
    arduino_interval = 1.0 / ARDUINO_SEND_FREQUENCY
    audio_interval = 1.0 / AUDIO_SEND_FREQUENCY
    
    last_arduino_send = 0
    last_audio_send = 0
    
    print(f"📡 数据发送线程启动:")
    print(f"   Arduino频率: {ARDUINO_SEND_FREQUENCY}Hz (间隔: {arduino_interval*1000:.1f}ms)")
    print(f"   音频频率: {AUDIO_SEND_FREQUENCY}Hz (间隔: {audio_interval*1000:.1f}ms)")
    
    arduino_count = 0
    audio_count = 0
    error_count = 0
    last_heartbeat = time.time()
    
    try:
        while system_running and not system_error:
            try:
                current_time = time.time()
                
                # 心跳检测 - 每10秒打印一次状态
                if current_time - last_heartbeat > 10:
                    print(f"💗 发送线程心跳: Arduino={arduino_count}, Audio={audio_count}, Errors={error_count}")
                    last_heartbeat = current_time
                    # 重置错误计数
                    if error_count > 50:  # 如果错误太多，标记系统错误
                        print("❌ 发送线程错误过多，标记系统错误")
                        system_error = True
                        break
                    error_count = 0
                
                # 获取最新数据（非阻塞）
                latest_data = None
                try:
                    while not gesture_data_queue.empty():
                        latest_data = gesture_data_queue.get_nowait()
                        gesture_data_queue.task_done()
                except:
                    pass
                
                if latest_data is None:
                    time.sleep(0.001)
                    continue
                
                # 发送数据到Arduino
                if current_time - last_arduino_send >= arduino_interval:
                    if mcu_connection:
                        try:
                            if send_to_arduino(mcu_connection, latest_data["normalized_angles"]):
                                arduino_count += 1
                            else:
                                error_count += 1
                        except Exception as e:
                            print(f"❌ Arduino发送异常: {e}")
                            error_count += 1
                    last_arduino_send = current_time
                
                # 发送数据到音频播放器
                if current_time - last_audio_send >= audio_interval:
                    try:
                        if send_to_audio_player(audio_process, latest_data["states"]):
                            audio_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        print(f"❌ 音频发送异常: {e}")
                        error_count += 1
                    last_audio_send = current_time
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"❌ 发送线程内部错误: {e}")
                error_count += 1
                time.sleep(0.1)
                
    except Exception as e:
        print(f"❌ 数据发送线程严重错误: {e}")
        print(traceback.format_exc())
        system_error = True
    
    print(f"📡 发送线程退出: Arduino={arduino_count}, Audio={audio_count}")

def main():
    global system_running, system_error
    
    print("🎵 模块化手势识别音乐系统 - 增强稳定性版")
    print("=" * 50)
    
    # 检查依赖
    if not check_and_install_dependencies():
        return
    
    # 设置音频系统
    audio_process = setup_audio_system()
    if not audio_process:
        print("❌ 音频系统设置失败，程序退出")
        return
    
    # 设置单片机连接
    mcu_connection = setup_mcu_connection()
    
    # 如果有单片机连接，启动串口监听线程
    serial_thread = None
    if mcu_connection:
        serial_thread = threading.Thread(target=serial_monitor, args=(mcu_connection,), daemon=True)
        serial_thread.start()
    
    # 创建手势数据队列
    import queue
    gesture_data_queue = queue.Queue(maxsize=5)  # 减小队列大小
    
    # 启动数据发送线程
    sender_thread = threading.Thread(
        target=data_sender_thread, 
        args=(audio_process, mcu_connection, gesture_data_queue), 
        daemon=True
    )
    sender_thread.start()
    
    # 主循环变量
    prevTime = 0
    frame_count = 0
    last_gc_time = time.time()
    last_status_time = time.time()
    
    # 滑动窗口
    WINDOW_SIZE = 3  # 减小窗口大小提高响应性
    angle_history = {
        "thumb": deque(maxlen=WINDOW_SIZE),
        "index": deque(maxlen=WINDOW_SIZE),
        "middle": deque(maxlen=WINDOW_SIZE),
        "ring": deque(maxlen=WINDOW_SIZE),
        "pinky": deque(maxlen=WINDOW_SIZE)
    }
    
    # 初始化窗口数据
    for i in range(WINDOW_SIZE):
        for finger in angle_history:
            angle_history[finger].append(0)
    
    # 当前手势状态
    current_gesture = {
        "angles": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0},
        "normalized_angles": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
        "states": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
    }

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率提高性能
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detector = HandDetector(maxHands=1, detectionCon=0.7)
        
        cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Control", 640, 600)
        
        print("\n🚀 系统启动完成！")
        print("📸 摄像头已就绪")
        print(f"🎵 音频播放已就绪 (接收频率: {AUDIO_SEND_FREQUENCY}Hz)")
        if mcu_connection:
            print(f"📡 Arduino串口通信已就绪 ({mcu_connection.connection.port}) (发送频率: {ARDUINO_SEND_FREQUENCY}Hz)")
        print("💡 弯曲手指即可播放音乐！")
        print("🎹 大拇指=do, 食指=re, 中指=mi, 无名指=sol, 小指=la")
        print("⏱️ 数据以固定频率稳定发送")
        print("❌ 按 'q' 退出程序")
        print("-" * 60)

        while system_running and not system_error:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                    
                frame_count += 1
                current_time = time.time()
                
                # 定期垃圾回收
                if current_time - last_gc_time > 30:  # 每30秒清理一次
                    gc.collect()
                    last_gc_time = current_time
                
                # 定期状态检查
                if current_time - last_status_time > 20:  # 每20秒检查一次
                    queue_size = gesture_data_queue.qsize()
                    print(f"💗 主线程心跳: Frame={frame_count}, Queue={queue_size}")
                    
                    # 检查音频进程状态
                    if audio_process and audio_process.poll() is not None:
                        print("⚠️ 音频进程已退出，标记系统错误")
                        system_error = True
                        break
                    
                    last_status_time = current_time
                
                # 手势识别
                frame = detector.findHands(frame)
                lmList = detector.findPosition(frame)
                
                current_angles, current_states = calculate_finger_angles_and_states(lmList)
                
                # 更新滑动窗口
                for finger in angle_history:
                    angle_history[finger].append(current_angles[finger])
                
                # 每N帧更新一次
                if frame_count % WINDOW_SIZE == 0:
                    # 计算平均角度
                    avg_angles = {}
                    for finger in angle_history:
                        avg_angles[finger] = sum(angle_history[finger]) / len(angle_history[finger])
                    
                    normalized_angles = normalize_angles_dict(avg_angles)
                    finger_states = current_states
                    
                    # 更新手势状态
                    current_gesture["angles"] = {k: round(v, 1) for k, v in avg_angles.items()}
                    current_gesture["normalized_angles"] = normalized_angles
                    current_gesture["states"] = finger_states
                    
                    # 将数据放入队列（非阻塞）
                    try:
                        gesture_data_queue.put_nowait(current_gesture.copy())
                    except queue.Full:
                        # 队列满时，清空队列并放入新数据
                        try:
                            while not gesture_data_queue.empty():
                                gesture_data_queue.get_nowait()
                                gesture_data_queue.task_done()
                            gesture_data_queue.put_nowait(current_gesture.copy())
                        except:
                            pass
                
                # 计算FPS
                if prevTime != 0:
                    fps = 1 / (current_time - prevTime)
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                prevTime = current_time
                
                # 显示角度信息
                y_offset = 60
                fingers = ["thumb", "index", "middle", "ring", "pinky"]
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
                for i, (finger, name) in enumerate(zip(fingers, finger_names)):
                    if finger in current_gesture["angles"]:
                        angle = current_gesture["angles"][finger]
                        normalized = current_gesture["normalized_angles"][finger]
                        state = current_gesture["states"][finger]
                        color = (0, 255, 0) if not state else (0, 0, 255)
                        
                        state_text = "Bent" if state else "Straight"
                        text = f"{name}: {angle:.1f}° (N:{normalized:.3f}) {state_text}"
                        
                        cv2.putText(frame, text, (10, y_offset + i * 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 显示系统状态
                status_y = y_offset + 5 * 25 + 10
                
                # 音频状态
                if audio_process and audio_process.poll() is None:
                    audio_status = f"Audio: {AUDIO_SEND_FREQUENCY}Hz - Running"
                    audio_color = (0, 255, 0)
                else:
                    audio_status = f"Audio: {AUDIO_SEND_FREQUENCY}Hz - Stopped"
                    audio_color = (0, 0, 255)
                cv2.putText(frame, audio_status, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, audio_color, 1)
                
                # Arduino状态
                if mcu_connection and mcu_connection.connection and mcu_connection.connection.is_open:
                    arduino_status = f"Arduino: {ARDUINO_SEND_FREQUENCY}Hz - Connected"
                    arduino_color = (0, 255, 0)
                else:
                    arduino_status = f"Arduino: {ARDUINO_SEND_FREQUENCY}Hz - Disconnected"
                    arduino_color = (0, 0, 255)
                cv2.putText(frame, arduino_status, (10, status_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, arduino_color, 1)
                
                # 队列状态
                queue_size = gesture_data_queue.qsize()
                queue_color = (0, 255, 0) if queue_size < 3 else (0, 255, 255) if queue_size < 5 else (0, 0, 255)
                cv2.putText(frame, f"Queue: {queue_size}/5", (10, status_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, queue_color, 1)
                
                # 系统状态
                if system_error:
                    cv2.putText(frame, "SYSTEM ERROR!", (10, status_y + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "System OK", (10, status_y + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                cv2.imshow("Hand Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            except Exception as e:
                print(f"❌ 主循环错误: {e}")
                print(traceback.format_exc())
                time.sleep(0.1)  # 错误后短暂暂停

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
                sender_thread.join(timeout=2)
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
                    audio_process.wait(timeout=3)
                    print("🎵 音频播放器已正常关闭")
                except subprocess.TimeoutExpired:
                    audio_process.kill()
                    audio_process.wait()
                    print("🎵 音频播放器已强制关闭")
            except Exception as e:
                print(f"⚠️ 关闭音频播放器时出错: {e}")
        
        # 清理队列
        if 'gesture_data_queue' in locals():
            try:
                while not gesture_data_queue.empty():
                    gesture_data_queue.get_nowait()
                    gesture_data_queue.task_done()
                print("📦 数据队列已清空")
            except:
                pass
        
        # 最终垃圾回收
        gc.collect()
        
        print("✅ 程序已完全退出")

if __name__ == "__main__":
    main()