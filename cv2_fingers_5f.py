import cv2
import mediapipe as mp # type: ignore
import time
import serial
import threading
import json
import math
import os
import sys
import subprocess
from collections import deque

# 定义每个手指的角度范围（根据实际测试调整）
FINGER_ANGLE_RANGES = {
    "thumb": {"min": 120, "max": 180},   # 拇指角度范围：120°=弯曲(0), 180°=伸直(1)
    "index": {"min": 5, "max": 180},     # 食指角度范围
    "middle": {"min": 5, "max": 180},    # 中指角度范围
    "ring": {"min": 5, "max": 180},      # 无名指角度范围
    "pinky": {"min": 5, "max": 180}      # 小指角度范围
}

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

def denormalize_angle(normalized_value, finger_name):
    """
    将归一化值转换回实际角度
    """
    if finger_name not in FINGER_ANGLE_RANGES:
        return 0.0
    
    min_angle = FINGER_ANGLE_RANGES[finger_name]["min"]
    max_angle = FINGER_ANGLE_RANGES[finger_name]["max"]
    
    # 限制在0-1范围内
    clamped_normalized = max(0.0, min(1.0, normalized_value))
    
    # 转换回实际角度
    actual_angle = min_angle + clamped_normalized * (max_angle - min_angle)
    
    return round(actual_angle, 1)

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

def calculate_angle(point1, point2, point3):
    """计算三个点之间的角度"""
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

def calculate_distance(point1, point2):
    """计算两点间距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_hand_left_or_right(lmList):
    """判断是左手还是右手"""
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

def calculate_thumb_improved(lmList):
    """改进的拇指检测算法"""
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
    
    # 方法1: 计算拇指各关节的角度
    angle_cmc_mcp_ip = calculate_angle(thumb_cmc, thumb_mcp, thumb_ip)
    angle_mcp_ip_tip = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
    
    # 方法2: 距离比较法
    # 当拇指弯曲时，拇指尖会更靠近手掌
    tip_to_wrist = calculate_distance(thumb_tip, wrist)
    tip_to_index = calculate_distance(thumb_tip, index_mcp)
    mcp_to_wrist = calculate_distance(thumb_mcp, wrist)
    
    # 方法3: 位置关系法
    # 建立手掌坐标系（以手腕到中指MCP为y轴）
    palm_vector = [middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]]
    thumb_vector = [thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1]]
    
    # 计算拇指向量在手掌方向的投影长度比例
    palm_length = math.sqrt(palm_vector[0]**2 + palm_vector[1]**2)
    if palm_length > 0:
        projection = (thumb_vector[0] * palm_vector[0] + thumb_vector[1] * palm_vector[1]) / palm_length
        projection_ratio = projection / palm_length
    else:
        projection_ratio = 0
    
    # 方法4: 拇指与食指的相对位置
    # 判断拇指是否向手掌内收
    hand_type = is_hand_left_or_right(lmList)
    
    # 计算拇指尖相对于拇指根部的横向偏移
    lateral_offset = thumb_tip[0] - thumb_mcp[0]
    
    # 根据左右手调整判断逻辑
    if hand_type == "right":
        # 右手：拇指向左收缩表示弯曲
        lateral_bent = lateral_offset > -20  # 调整阈值
    else:
        # 左手：拇指向右收缩表示弯曲
        lateral_bent = lateral_offset < 20   # 调整阈值
    
    # 综合判断逻辑
    detection_scores = []
    
    # 评分1: 关节角度评分（权重0.3）
    angle_score = 0
    if angle_cmc_mcp_ip < 160:  # 第一关节弯曲
        angle_score += 0.5
    if angle_mcp_ip_tip < 160:  # 第二关节弯曲
        angle_score += 0.5
    detection_scores.append(angle_score)
    
    # 评分2: 距离评分（权重0.25）
    distance_score = 0
    if mcp_to_wrist > 0:
        tip_wrist_ratio = tip_to_wrist / mcp_to_wrist
        if tip_wrist_ratio < 1.3:  # 拇指尖相对更接近手腕
            distance_score += 0.5
    
    if tip_to_index < 60:  # 拇指尖接近食指根部
        distance_score += 0.5
    detection_scores.append(distance_score)
    
    # 评分3: 投影评分（权重0.25）
    projection_score = 1.0 if projection_ratio < 0.5 else 0.0
    detection_scores.append(projection_score)
    
    # 评分4: 横向位置评分（权重0.2）
    lateral_score = 1.0 if lateral_bent else 0.0
    detection_scores.append(lateral_score)
    
    # 加权平均
    weights = [0.3, 0.25, 0.25, 0.2]
    final_score = sum(score * weight for score, weight in zip(detection_scores, weights))
    
    # 判断弯曲状态
    is_bent = final_score > 0.4  # 降低阈值，提高敏感度
    
    # 计算综合角度（用于显示）
    main_angle = (angle_cmc_mcp_ip + angle_mcp_ip_tip) / 2
    
    return main_angle, is_bent

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
    
    if len(lmList) < 21:
        return angles, states
    
    # 拇指特殊处理 - 使用改进的检测算法
    if len(lmList) >= 5:
        angles["thumb"], states["thumb"] = calculate_thumb_improved(lmList)
    
    # 食指
    if len(lmList) >= 9:
        angles["index"] = calculate_angle(
            [lmList[5][1], lmList[5][2]],  # MCP
            [lmList[6][1], lmList[6][2]],  # PIP
            [lmList[8][1], lmList[8][2]]   # TIP
        )
        # 食指弯曲判断：角度判断
        states["index"] = angles["index"] < 50
    
    # 中指
    if len(lmList) >= 13:
        angles["middle"] = calculate_angle(
            [lmList[9][1], lmList[9][2]],   # MCP
            [lmList[10][1], lmList[10][2]], # PIP
            [lmList[12][1], lmList[12][2]]  # TIP
        )
        # 中指弯曲判断：角度判断
        states["middle"] = angles["middle"] < 50
    
    # 无名指
    if len(lmList) >= 17:
        angles["ring"] = calculate_angle(
            [lmList[13][1], lmList[13][2]], # MCP
            [lmList[14][1], lmList[14][2]], # PIP
            [lmList[16][1], lmList[16][2]]  # TIP
        )
        # 无名指弯曲判断：角度判断
        states["ring"] =  angles["ring"] < 60
    
    # 小指
    if len(lmList) >= 21:
        angles["pinky"] = calculate_angle(
            [lmList[17][1], lmList[17][2]], # MCP
            [lmList[18][1], lmList[18][2]], # PIP
            [lmList[20][1], lmList[20][2]]  # TIP
        )
        # 小指弯曲判断：角度判断
        states["pinky"] = angles["pinky"] < 120
    
    return angles, states

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
    """设置音频系统：生成音频文件并启动播放器"""
    print("🎵 设置音频系统...")
    
    # 检查是否需要生成音频文件
    if not os.path.exists("sounds") or len([f for f in os.listdir("sounds") if f.endswith('.wav')]) < 5:
        print("📝 生成音频文件...")
        try:
            result = subprocess.run([sys.executable, "five_tones.py"], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"❌ 音频生成失败: {result.stderr}")
                return None
            print("✅ 音频文件生成完成")
        except subprocess.TimeoutExpired:
            print("❌ 音频生成超时")
            return None
        except FileNotFoundError:
            print("❌ 找不到 five_tones.py 文件")
            return None
    else:
        print("✅ 音频文件已存在")
    
    # 启动音频播放器进程
    print("🎧 启动音频播放器...")
    try:
        audio_process = subprocess.Popen(
            [sys.executable, "audio_player.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 给播放器一点时间初始化
        time.sleep(2)
        
        if audio_process.poll() is None:  # 进程仍在运行
            print("✅ 音频播放器启动成功")
            return audio_process
        else:
            print("❌ 音频播放器启动失败")
            return None
            
    except FileNotFoundError:
        print("❌ 找不到 audio_player.py 文件")
        return None

def send_to_audio_player(audio_process, gesture_data):
    """发送手势数据到音频播放器"""
    if audio_process and audio_process.stdin:
        try:
            json_msg = json.dumps(gesture_data, separators=(',', ':'))
            audio_process.stdin.write(json_msg + '\n')
            audio_process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            print("⚠️ 音频播放器连接断开")
            return False
    return False

def main():
    print("🎵 模块化手势识别音乐系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_and_install_dependencies():
        return
    
    # 设置音频系统
    audio_process = setup_audio_system()
    if not audio_process:
        print("❌ 音频系统设置失败，程序退出")
        return
    
    # 设置串口
    ser = None
    try:
        ser = serial.Serial(
            port="COM24",
            baudrate=9600,
            timeout=0.1,
            write_timeout=1
        )
        print(f"✅ 串口 {ser.port} 连接成功")
        
        # 启动串口监听线程
        serial_thread = threading.Thread(target=serial_monitor, args=(ser,), daemon=True)
        serial_thread.start()
        
    except serial.SerialException as e:
        print(f"⚠️ 串口连接失败: {e} (将继续运行，但不发送数据到Arduino)")
    
    # 主要的手势识别逻辑
    prevTime = 0
    frame_count = 0
    
    # 创建滑动窗口存储最近5帧的角度数据
    WINDOW_SIZE = 5
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
        "timestamp": 0,
        "angles": {
            "thumb": 0,
            "index": 0,
            "middle": 0,
            "ring": 0,
            "pinky": 0
        },
        "normalized_angles": {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "pinky": 0.0
        },
        "states": {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
    }

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        detector = HandDetector(maxHands=1, detectionCon=0.7)
        
        cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Control", 800, 600)
        
        print("\n🚀 系统启动完成！")
        print("📸 摄像头已就绪")
        print("🎵 音频播放已就绪")
        if ser:
            print("📡 Arduino串口通信已就绪")
        print("💡 弯曲手指即可播放音乐！")
        print("🎹 大拇指=do, 食指=re, 中指=mi, 无名指=sol, 小指=la")
        print("❌ 按 'q' 退出程序")
        print("-" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
                
            frame_count += 1
            
            # 始终检测手部并绘制关键点
            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame)
            
            # 计算当前帧的角度和状态
            current_angles, current_states = calculate_finger_angles_and_states(lmList)
            
            # 更新滑动窗口
            for finger in angle_history:
                angle_history[finger].append(current_angles[finger])
            
            # 每5帧计算一次平均值并发送数据
            if frame_count % WINDOW_SIZE == 0:
                # 计算平均角度
                avg_angles = {}
                for finger in angle_history:
                    avg_angles[finger] = sum(angle_history[finger]) / len(angle_history[finger])
                
                # 归一化角度
                normalized_angles = normalize_angles_dict(avg_angles)
                
                # 使用改进的状态判断（结合角度和位置）
                finger_states = current_states
                
                # 检查是否有状态变化
                change = False
                for finger in avg_angles:
                    if (abs(avg_angles[finger] - current_gesture["angles"][finger]) > 5 or 
                        finger_states[finger] != current_gesture["states"][finger]):
                        change = True
                        break
                
                if change:
                    # 更新当前手势状态
                    current_gesture["timestamp"] = int(time.time() * 1000)
                    current_gesture["angles"] = {k: round(v, 1) for k, v in avg_angles.items()}
                    current_gesture["normalized_angles"] = normalized_angles
                    current_gesture["states"] = finger_states
                    
                    # 发送到音频播放器
                    if not send_to_audio_player(audio_process, current_gesture):
                        print("⚠️ 音频播放器连接中断，尝试重启...")
                        # 可以在这里添加重启音频播放器的逻辑
                    
                    # 发送到Arduino（如果连接）
                    if ser and ser.is_open:
                        try:
                            json_msg = json.dumps(current_gesture, separators=(',', ':')) + '\n'
                            ser.write(json_msg.encode('utf-8'))
                            ser.flush()
                        except serial.SerialException as e:
                            print(f"❌ 串口发送错误: {e}")
            
            # 计算并显示FPS
            currentTime = time.time()
            if prevTime != 0:
                fps = 1 / (currentTime - prevTime)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            prevTime = currentTime
            
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
                    
                    # 显示原始角度、归一化值和状态
                    state_text = "Bent" if state else "Straight"
                    text = f"{name}: {angle:.1f}° (N:{normalized:.3f}) {state_text}"
                    
                    cv2.putText(frame, text, (10, y_offset + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 显示音频播放器状态
            audio_status = "🎵 Audio: Running" if audio_process and audio_process.poll() is None else "❌ Audio: Stopped"
            cv2.putText(frame, audio_status, (10, y_offset + 5 * 30 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    
    finally:
        # 清理资源
        if 'cap' in locals():
            cap.release()
        
        if ser and ser.is_open:
            ser.close()
            print("📡 串口已关闭")
        
        # 关闭音频播放器进程
        if audio_process:
            try:
                audio_process.stdin.close()
                audio_process.terminate()
                audio_process.wait(timeout=5)
                print("🎵 音频播放器已关闭")
            except subprocess.TimeoutExpired:
                audio_process.kill()
                print("🎵 强制关闭音频播放器")
        
        cv2.destroyAllWindows()
        print("✅ 程序已完全退出")

if __name__ == "__main__":
    main()