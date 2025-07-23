import serial
import serial.tools.list_ports
import time
import re

class MicrocontrollerConnection:
    def __init__(self, baudrate=9600, timeout=1):
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        
        # 常见单片机的识别模式
        self.mcu_patterns = [
            r'Arduino',
            r'CH340',
            r'CP210',
            r'FT232',
            r'USB-SERIAL',
            r'STM32',
            r'ESP32',
            r'ESP8266'
        ]
    
    def find_mcu_ports(self):
        """查找可能的单片机串口"""
        ports = serial.tools.list_ports.comports()
        mcu_ports = []
        
        for port in ports:
            port_info = {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'manufacturer': getattr(port, 'manufacturer', 'Unknown'),
                'score': 0  # 匹配得分
            }
            
            # 根据描述和硬件ID评分
            text_to_check = f"{port.description} {port.hwid}".upper()
            
            for pattern in self.mcu_patterns:
                if re.search(pattern.upper(), text_to_check):
                    port_info['score'] += 10
            
            # 额外加分条件
            if 'USB' in text_to_check:
                port_info['score'] += 5
            if 'SERIAL' in text_to_check:
                port_info['score'] += 3
                
            mcu_ports.append(port_info)
        
        # 按得分排序，优先尝试得分高的端口
        mcu_ports.sort(key=lambda x: x['score'], reverse=True)
        return mcu_ports
    
    def test_connection(self, port_device, test_commands=['ping\n', 'AT\n', '\n']):
        """测试串口连接"""
        try:
            ser = serial.Serial(
                port=port_device,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            time.sleep(0.5)  # 等待连接稳定
            
            # 清空输入缓冲区
            ser.reset_input_buffer()
            
            # 尝试发送测试命令
            for cmd in test_commands:
                ser.write(cmd.encode())
                time.sleep(0.2)
                
                if ser.in_waiting > 0:
                    response = ser.read(ser.in_waiting)
                    print(f"收到响应: {response}")
                    return ser  # 有响应，认为连接成功
            
            # 即使没有响应，也可能是单片机不回应ping命令
            print(f"连接到 {port_device}，但无响应（可能正常）")
            return ser
            
        except Exception as e:
            print(f"连接 {port_device} 失败: {e}")
            return None
    
    def auto_connect(self):
        """自动连接单片机"""
        mcu_ports = self.find_mcu_ports()
        
        if not mcu_ports:
            print("未找到任何串口设备")
            return False
        
        print("检测到的串口设备（按可能性排序）：")
        for i, port in enumerate(mcu_ports):
            print(f"{i+1}. {port['device']} - {port['description']} (得分: {port['score']})")
        
        # 尝试连接，优先尝试得分高的
        for port in mcu_ports:
            if port['score'] > 0:  # 只尝试有可能是单片机的端口
                print(f"\n尝试连接到 {port['device']}...")
                connection = self.test_connection(port['device'])
                
                if connection:
                    self.connection = connection
                    print(f"✓ 成功连接到 {port['device']}")
                    return True
        
        # 如果没有高分端口成功，尝试所有端口
        print("\n尝试连接其他端口...")
        for port in mcu_ports:
            if port['score'] == 0:
                print(f"尝试连接到 {port['device']}...")
                connection = self.test_connection(port['device'])
                
                if connection:
                    self.connection = connection
                    print(f"✓ 成功连接到 {port['device']}")
                    return True
        
        print("❌ 无法连接到任何设备")
        return False
    
    def send(self, data):
        """发送数据"""
        if not self.connection or not self.connection.is_open:
            print("连接无效")
            return False
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            self.connection.write(data)
            print(f"✓ 已发送: {data}")
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def receive(self, timeout=1):
        """接收数据"""
        if not self.connection or not self.connection.is_open:
            return None
        
        try:
            old_timeout = self.connection.timeout
            self.connection.timeout = timeout
            
            if self.connection.in_waiting > 0:
                data = self.connection.read(self.connection.in_waiting)
                return data.decode('utf-8', errors='ignore')
            
            return None
        except Exception as e:
            print(f"接收数据失败: {e}")
            return None
        finally:
            self.connection.timeout = old_timeout
    
    def close(self):
        """关闭连接"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("串口连接已关闭")

def main():
    """使用示例"""
    # 创建连接对象
    mcu = MicrocontrollerConnection(baudrate=9600)
    
    # 自动连接
    if mcu.auto_connect():
        try:
            # 发送数据
            mcu.send("Hello from Python!\n")
            
            # 等待并接收回复
            time.sleep(0.5)
            response = mcu.receive(timeout=2)
            if response:
                print(f"收到回复: {response.strip()}")
            
            # 交互式通信
            print("\n=== 开始交互式通信 ===")
            print("输入 'quit' 退出")
            
            while True:
                user_input = input("发送数据: ")
                if user_input.lower() == 'quit':
                    break
                
                mcu.send(user_input + '\n')
                time.sleep(0.1)
                
                response = mcu.receive(timeout=1)
                if response:
                    print(f"回复: {response.strip()}")
                
        finally:
            mcu.close()
    else:
        print("无法建立连接")

if __name__ == "__main__":
    main()