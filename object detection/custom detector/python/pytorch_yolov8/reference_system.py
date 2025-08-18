#!/usr/bin/env python3
"""
透视参考系标定脚本
用于标定六个关键点并建立透视参考系，生成JSON配置文件
供排球计数系统使用
"""

import numpy as np
import cv2
import pyzed.sl as sl
import json
import os
from enum import Enum

class CalibrationState(Enum):
    """标定状态枚举"""
    WAITING = "Waiting to start"
    CALIBRATING_A_LEFT = "Calibrating Bar A Left Point"
    CALIBRATING_A_RIGHT = "Calibrating Bar A Right Point"
    CALIBRATING_B_LEFT = "Calibrating Bar B Left Point"
    CALIBRATING_B_RIGHT = "Calibrating Bar B Right Point"
    CALIBRATING_C_LEFT = "Calibrating Bar C Left Point"
    CALIBRATING_C_RIGHT = "Calibrating Bar C Right Point"
    COMPLETED = "Calibration Completed"

class PerspectiveCalibrator:
    """透视参考系标定器"""
    def __init__(self):
        # 相机和栏杆参数
        self.camera_height = 1.5  # 相机高度（米）
        self.bar_height = 2.35    # 栏杆高度（米）
        
        # 栏杆距离（米）
        self.bar_distances = {
            'A': 3.5,
            'B': 5.0,
            'C': 6.5
        }
        
        # 标定点存储
        self.bar_2d_points = {
            'A_left': None,
            'A_right': None,
            'B_left': None,
            'B_right': None,
            'C_left': None,
            'C_right': None
        }
        
        # 距离到Y像素的映射
        self.distance_to_y_pixel = {}
        
        # 标定状态
        self.current_state = CalibrationState.WAITING
        self.is_calibrated = False
        
        # 鼠标回调参数
        self.current_image = None
        self.display_image = None
        self.mouse_x = 0
        self.mouse_y = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_point_selection(x, y)
    
    def handle_point_selection(self, x, y):
        """处理点选择"""
        if self.current_state == CalibrationState.CALIBRATING_A_LEFT:
            self.bar_2d_points['A_left'] = (x, y)
            print(f"栏杆A左侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_A_RIGHT
            
        elif self.current_state == CalibrationState.CALIBRATING_A_RIGHT:
            self.bar_2d_points['A_right'] = (x, y)
            print(f"栏杆A右侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_B_LEFT
            
        elif self.current_state == CalibrationState.CALIBRATING_B_LEFT:
            self.bar_2d_points['B_left'] = (x, y)
            print(f"栏杆B左侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_B_RIGHT
            
        elif self.current_state == CalibrationState.CALIBRATING_B_RIGHT:
            self.bar_2d_points['B_right'] = (x, y)
            print(f"栏杆B右侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_C_LEFT
            
        elif self.current_state == CalibrationState.CALIBRATING_C_LEFT:
            self.bar_2d_points['C_left'] = (x, y)
            print(f"栏杆C左侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_C_RIGHT
            
        elif self.current_state == CalibrationState.CALIBRATING_C_RIGHT:
            self.bar_2d_points['C_right'] = (x, y)
            print(f"栏杆C右侧点标定: ({x}, {y})")
            self.current_state = CalibrationState.COMPLETED
            print("\n六个点标定完成！开始建立参考系...")
            self.build_reference_system()
    
    def build_reference_system(self):
        """建立参考系 - 使用线性插值计算中间距离的Y坐标"""
        print("\n建立距离-Y像素映射参考系...")
        
        # 获取三个栏杆的Y坐标（使用左右点的平均值）
        y_A = (self.bar_2d_points['A_left'][1] + self.bar_2d_points['A_right'][1]) / 2
        y_B = (self.bar_2d_points['B_left'][1] + self.bar_2d_points['B_right'][1]) / 2
        y_C = (self.bar_2d_points['C_left'][1] + self.bar_2d_points['C_right'][1]) / 2
        
        # 添加三个栏杆的基准点
        self.distance_to_y_pixel[3.5] = y_A
        self.distance_to_y_pixel[5.0] = y_B
        self.distance_to_y_pixel[6.5] = y_C
        
        # 计算3.5米到5米之间的Y坐标（不包括端点）
        print("\n计算3.5米到5米之间的参考线...")
        for i in range(1, 15):  # 3.6, 3.7, ... 4.9
            distance = 3.5 + i * 0.1
            # 线性插值
            ratio = (distance - 3.5) / (5.0 - 3.5)
            y_interpolated = y_A + ratio * (y_B - y_A)
            self.distance_to_y_pixel[round(distance, 1)] = y_interpolated
            print(f"  距离 {distance:.1f}m -> Y像素: {y_interpolated:.1f}")
        
        # 计算5米到6.5米之间的Y坐标（不包括端点）
        print("\n计算5米到6.5米之间的参考线...")
        for i in range(1, 15):  # 5.1, 5.2, ... 6.4
            distance = 5.0 + i * 0.1
            # 线性插值
            ratio = (distance - 5.0) / (6.5 - 5.0)
            y_interpolated = y_B + ratio * (y_C - y_B)
            self.distance_to_y_pixel[round(distance, 1)] = y_interpolated
            print(f"  距离 {distance:.1f}m -> Y像素: {y_interpolated:.1f}")
        
        self.is_calibrated = True
        print(f"\n参考系建立完成！共生成 {len(self.distance_to_y_pixel)} 个距离映射点")
    
    def save_calibration(self, filename="perspective_calibration.json"):
        """保存标定数据到JSON文件"""
        if not self.is_calibrated:
            print("标定未完成，无法保存")
            return False
        
        calibration_data = {
            'camera_height': self.camera_height,
            'bar_height': self.bar_height,
            'bar_distances': self.bar_distances,
            'bar_2d_points': {
                k: list(v) if v is not None else None 
                for k, v in self.bar_2d_points.items()
            },
            'distance_to_y_pixel': {
                str(k): v for k, v in self.distance_to_y_pixel.items()
            },
            'is_calibrated': self.is_calibrated
        }
        try:
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"\n标定数据已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def draw_calibration_ui(self):
        """绘制标定界面"""
        if self.current_image is None:
            return
        self.display_image = self.current_image.copy()
        # 绘制已标定的点
        colors = {
            'A': (0, 255, 255),  # 黄色
            'B': (255, 255, 0),  # 青色
            'C': (255, 0, 255)   # 品红色
        }
        # 绘制已标定的点和连线
        for bar in ['A', 'B', 'C']:
            left_key = f'{bar}_left'
            right_key = f'{bar}_right'
            
            if self.bar_2d_points[left_key]:
                cv2.circle(self.display_image, self.bar_2d_points[left_key], 
                          8, colors[bar], -1)
                cv2.putText(self.display_image, f"{bar}L", 
                           (self.bar_2d_points[left_key][0] - 20, 
                            self.bar_2d_points[left_key][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[bar], 2)
            
            if self.bar_2d_points[right_key]:
                cv2.circle(self.display_image, self.bar_2d_points[right_key], 
                          8, colors[bar], -1)
                cv2.putText(self.display_image, f"{bar}R", 
                           (self.bar_2d_points[right_key][0] - 20, 
                            self.bar_2d_points[right_key][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[bar], 2)
            
            # 如果左右点都已标定，绘制连线
            if self.bar_2d_points[left_key] and self.bar_2d_points[right_key]:
                cv2.line(self.display_image, 
                        self.bar_2d_points[left_key], 
                        self.bar_2d_points[right_key], 
                        colors[bar], 2)
        
        # 绘制鼠标十字线（仅在标定过程中）
        if self.current_state not in [CalibrationState.WAITING, CalibrationState.COMPLETED]:
            cv2.line(self.display_image, (self.mouse_x, 0), 
                    (self.mouse_x, self.display_image.shape[0]), (0, 255, 0), 1)
            cv2.line(self.display_image, (0, self.mouse_y), 
                    (self.display_image.shape[1], self.mouse_y), (0, 255, 0), 1)
        
        # 绘制状态信息
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (10, 10), (600, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, self.display_image, 0.7, 0, self.display_image)
        
        # 显示状态文本
        cv2.putText(self.display_image, f"Status: {self.current_state.value}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示操作提示
        if self.current_state == CalibrationState.WAITING:
            cv2.putText(self.display_image, "Press 'C' to start calibration", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif self.current_state == CalibrationState.COMPLETED:
            cv2.putText(self.display_image, "Calibration completed! Press 'S' to save, 'R' to recalibrate", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(self.display_image, "Click points to calibrate", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(self.display_image, "Press 'ESC' to exit", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def reset_calibration(self):
        """重置标定"""
        self.bar_2d_points = {
            'A_left': None,
            'A_right': None,
            'B_left': None,
            'B_right': None,
            'C_left': None,
            'C_right': None
        }
        self.distance_to_y_pixel = {}
        self.current_state = CalibrationState.WAITING
        self.is_calibrated = False
        print("\n标定已重置")

def main():
    """主函数"""
    print("="*60)
    print("透视参考系标定程序")
    print("="*60)
    
    # 初始化相机
    print("\n初始化ZED相机...")
    zed = sl.Camera()
    
    # 设置初始化参数
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 标定时不需要深度
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    
    # 打开相机
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败: {repr(status)}")
        exit(1)
    
    print("相机初始化成功")
    
    # 获取相机信息
    camera_info = zed.get_camera_information()
    camera_res = camera_info.camera_configuration.resolution
    print(f"相机分辨率: {camera_res.width}x{camera_res.height}")
    
    # 创建图像容器
    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    # 创建标定器
    calibrator = PerspectiveCalibrator()
    
    # 创建窗口名称
    window_name = "Reference System Calibration"
    
    # 先获取一帧图像用于初始化窗口
    print("\n初始化显示窗口...")
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # 直接获取原始分辨率的左目图像
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
        
        # 转换为OpenCV格式
        image_ocv = image_zed.get_data()
        if image_ocv is not None:
            # BGR转换（ZED返回BGRA格式）
            initial_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            
            # 创建窗口并显示初始图像
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, camera_res.width, camera_res.height)
            cv2.imshow(window_name, initial_image)
            cv2.waitKey(1)  # 确保窗口完全创建
            
            # 设置鼠标回调
            cv2.setMouseCallback(window_name, calibrator.mouse_callback)
            print("窗口初始化成功")
    else:
        print("无法获取初始图像，请检查相机连接")
        zed.close()
        exit(1)
    
    print("\n标定说明:")
    print("-"*40)
    print("1. 按 'C' 键开始标定")
    print("2. 依次点击六个标定点:")
    print("   - 栏杆A (3.5m) 左侧点")
    print("   - 栏杆A (3.5m) 右侧点")
    print("   - 栏杆B (5.0m) 左侧点")
    print("   - 栏杆B (5.0m) 右侧点")
    print("   - 栏杆C (6.5m) 左侧点")
    print("   - 栏杆C (6.5m) 右侧点")
    print("3. 标定完成后按 'S' 保存")
    print("4. 按 'R' 重新标定")
    print("5. 按 'ESC' 退出")
    print("-"*40)
    
    try:
        while True:
            # 获取图像
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 获取原始分辨率的左目图像
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
                
                # 转换为OpenCV格式
                image_ocv = image_zed.get_data()
                if image_ocv is not None:
                    # BGR转换（ZED返回BGRA格式）
                    calibrator.current_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
                    
                    # 绘制UI
                    calibrator.draw_calibration_ui()
                    
                    # 显示图像
                    if calibrator.display_image is not None:
                        cv2.imshow(window_name, calibrator.display_image)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n退出程序")
                break
                
            elif key == ord('c') or key == ord('C'):
                if calibrator.current_state == CalibrationState.WAITING:
                    # 截取当前帧用于标定
                    print("\n开始标定，请依次点击六个标定点")
                    calibrator.current_state = CalibrationState.CALIBRATING_A_LEFT
                    
            elif key == ord('s') or key == ord('S'):
                if calibrator.current_state == CalibrationState.COMPLETED:
                    calibrator.save_calibration()
                    
            elif key == ord('r') or key == ord('R'):
                calibrator.reset_calibration()
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        
    finally:
        # 清理资源
        zed.close()
        cv2.destroyAllWindows()
        print("程序退出")

if __name__ == "__main__":
    main()