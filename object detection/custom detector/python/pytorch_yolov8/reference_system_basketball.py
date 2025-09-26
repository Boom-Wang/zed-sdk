#!/usr/bin/env python3
"""
篮球投球ROI标定系统
用于标定篮板、篮筐上方区域和篮网的ROI区域
生成JSON配置文件供篮球投球计数系统使用
"""

import numpy as np
import cv2
import pyzed.sl as sl
import json
import os
from enum import Enum
from typing import Dict, Tuple, List, Optional

class CalibrationState(Enum):
    """标定状态枚举"""
    WAITING = "Waiting to start"
    # 篮板ROI标定状态
    CALIBRATING_BACKBOARD_TL = "Calibrating Backboard Top-Left"
    CALIBRATING_BACKBOARD_BL = "Calibrating Backboard Bottom-Left"
    CALIBRATING_BACKBOARD_BR = "Calibrating Backboard Bottom-Right"
    CALIBRATING_BACKBOARD_TR = "Calibrating Backboard Top-Right"
    # 篮筐上方ROI标定状态
    CALIBRATING_ABOVE_HOOP_TL = "Calibrating Above Hoop Top-Left"
    CALIBRATING_ABOVE_HOOP_BL = "Calibrating Above Hoop Bottom-Left"
    CALIBRATING_ABOVE_HOOP_BR = "Calibrating Above Hoop Bottom-Right"
    CALIBRATING_ABOVE_HOOP_TR = "Calibrating Above Hoop Top-Right"
    # 篮网ROI标定状态
    CALIBRATING_NET_TL = "Calibrating Net Top-Left"
    CALIBRATING_NET_BL = "Calibrating Net Bottom-Left"
    CALIBRATING_NET_BR = "Calibrating Net Bottom-Right"
    CALIBRATING_NET_TR = "Calibrating Net Top-Right"
    COMPLETED = "Calibration Completed"

class BasketballROICalibrator:
    """篮球ROI标定器"""
    
    def __init__(self):
        """初始化标定器"""
        # ROI点存储 - 每个ROI存储四个角点
        self.roi_points = {
            'backboard': {
                'top_left': None,
                'bottom_left': None,
                'bottom_right': None,
                'top_right': None
            },
            'above_hoop': {
                'top_left': None,
                'bottom_left': None,
                'bottom_right': None,
                'top_right': None
            },
            'net': {
                'top_left': None,
                'bottom_left': None,
                'bottom_right': None,
                'top_right': None
            }
        }
        
        # ROI颜色定义
        self.roi_colors = {
            'backboard': (0, 255, 255),    # 黄色
            'above_hoop': (0, 255, 0),      # 绿色
            'net': (0, 0, 255)              # 红色
        }
        
        # 标定状态
        self.current_state = CalibrationState.WAITING
        self.is_calibrated = False
        
        # 鼠标回调参数
        self.current_image = None
        self.display_image = None
        self.mouse_x = 0
        self.mouse_y = 0
        
        # 窗口名称
        self.window_name = "Basketball ROI Calibration"
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_point_selection(x, y)
    
    def handle_point_selection(self, x, y):
        """处理点选择"""
        # 篮板ROI标定
        if self.current_state == CalibrationState.CALIBRATING_BACKBOARD_TL:
            self.roi_points['backboard']['top_left'] = (x, y)
            print(f"篮板左上角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_BACKBOARD_BL
            
        elif self.current_state == CalibrationState.CALIBRATING_BACKBOARD_BL:
            self.roi_points['backboard']['bottom_left'] = (x, y)
            print(f"篮板左下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_BACKBOARD_BR
            
        elif self.current_state == CalibrationState.CALIBRATING_BACKBOARD_BR:
            self.roi_points['backboard']['bottom_right'] = (x, y)
            print(f"篮板右下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_BACKBOARD_TR
            
        elif self.current_state == CalibrationState.CALIBRATING_BACKBOARD_TR:
            self.roi_points['backboard']['top_right'] = (x, y)
            print(f"篮板右上角点: ({x}, {y})")
            print("篮板ROI标定完成！\n")
            self.current_state = CalibrationState.CALIBRATING_ABOVE_HOOP_TL
            
        # 篮筐上方ROI标定
        elif self.current_state == CalibrationState.CALIBRATING_ABOVE_HOOP_TL:
            self.roi_points['above_hoop']['top_left'] = (x, y)
            print(f"篮筐上方左上角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_ABOVE_HOOP_BL
            
        elif self.current_state == CalibrationState.CALIBRATING_ABOVE_HOOP_BL:
            self.roi_points['above_hoop']['bottom_left'] = (x, y)
            print(f"篮筐上方左下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_ABOVE_HOOP_BR
            
        elif self.current_state == CalibrationState.CALIBRATING_ABOVE_HOOP_BR:
            self.roi_points['above_hoop']['bottom_right'] = (x, y)
            print(f"篮筐上方右下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_ABOVE_HOOP_TR
            
        elif self.current_state == CalibrationState.CALIBRATING_ABOVE_HOOP_TR:
            self.roi_points['above_hoop']['top_right'] = (x, y)
            print(f"篮筐上方右上角点: ({x}, {y})")
            print("篮筐上方ROI标定完成！\n")
            self.current_state = CalibrationState.CALIBRATING_NET_TL
            
        # 篮网ROI标定
        elif self.current_state == CalibrationState.CALIBRATING_NET_TL:
            self.roi_points['net']['top_left'] = (x, y)
            print(f"篮网左上角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_NET_BL
            
        elif self.current_state == CalibrationState.CALIBRATING_NET_BL:
            self.roi_points['net']['bottom_left'] = (x, y)
            print(f"篮网左下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_NET_BR
            
        elif self.current_state == CalibrationState.CALIBRATING_NET_BR:
            self.roi_points['net']['bottom_right'] = (x, y)
            print(f"篮网右下角点: ({x}, {y})")
            self.current_state = CalibrationState.CALIBRATING_NET_TR
            
        elif self.current_state == CalibrationState.CALIBRATING_NET_TR:
            self.roi_points['net']['top_right'] = (x, y)
            print(f"篮网右上角点: ({x}, {y})")
            print("篮网ROI标定完成！\n")
            self.current_state = CalibrationState.COMPLETED
            self.is_calibrated = True
            print("所有ROI标定完成！")
    
    def draw_roi(self, image, roi_name, color, thickness=2):
        """绘制单个ROI区域"""
        roi = self.roi_points[roi_name]
        
        # 检查所有点是否都已标定
        if all(point is not None for point in roi.values()):
            # 获取四个角点
            pts = np.array([
                roi['top_left'],
                roi['top_right'],
                roi['bottom_right'],
                roi['bottom_left']
            ], np.int32)
            
            # 绘制多边形轮廓
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
            
            # 绘制角点
            for point in roi.values():
                if point is not None:
                    cv2.circle(image, point, 5, color, -1)
    
    def draw_calibration_ui(self):
        """绘制标定界面"""
        if self.current_image is None:
            return
            
        self.display_image = self.current_image.copy()
        
        # 绘制已标定的ROI
        self.draw_roi(self.display_image, 'backboard', self.roi_colors['backboard'])
        self.draw_roi(self.display_image, 'above_hoop', self.roi_colors['above_hoop'])
        self.draw_roi(self.display_image, 'net', self.roi_colors['net'])
        
        # 绘制正在标定的点
        current_roi = None
        current_corner = None
        
        if 'BACKBOARD' in self.current_state.name:
            current_roi = 'backboard'
        elif 'ABOVE_HOOP' in self.current_state.name:
            current_roi = 'above_hoop'
        elif 'NET' in self.current_state.name:
            current_roi = 'net'
        
        # 绘制已标定但未完成的ROI的点
        if current_roi:
            roi = self.roi_points[current_roi]
            color = self.roi_colors[current_roi]
            for corner_name, point in roi.items():
                if point is not None:
                    cv2.circle(self.display_image, point, 5, color, -1)
                    # 添加标签
                    label = corner_name.replace('_', ' ').title()
                    cv2.putText(self.display_image, label[:2], 
                               (point[0] - 10, point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制鼠标十字线（仅在标定过程中）
        if self.current_state not in [CalibrationState.WAITING, CalibrationState.COMPLETED]:
            cv2.line(self.display_image, (self.mouse_x, 0), 
                    (self.mouse_x, self.display_image.shape[0]), (255, 255, 255), 1)
            cv2.line(self.display_image, (0, self.mouse_y), 
                    (self.display_image.shape[1], self.mouse_y), (255, 255, 255), 1)
        
        # 绘制状态信息背景
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (10, 10), (800, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, self.display_image, 0.7, 0, self.display_image)
        
        # 显示状态文本
        cv2.putText(self.display_image, f"Status: {self.current_state.value}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示ROI标识
        cv2.rectangle(self.display_image, (20, 60), (40, 80), self.roi_colors['backboard'], -1)
        cv2.putText(self.display_image, "Backboard ROI", (50, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(self.display_image, (220, 60), (240, 80), self.roi_colors['above_hoop'], -1)
        cv2.putText(self.display_image, "Above Hoop ROI", (250, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(self.display_image, (440, 60), (460, 80), self.roi_colors['net'], -1)
        cv2.putText(self.display_image, "Net ROI", (470, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示操作提示
        if self.current_state == CalibrationState.WAITING:
            cv2.putText(self.display_image, "Press 'C' to start calibration", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif self.current_state == CalibrationState.COMPLETED:
            cv2.putText(self.display_image, "Calibration completed! Press 'S' to save, 'R' to recalibrate", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(self.display_image, "Click to set corner points (TL -> BL -> BR -> TR)", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(self.display_image, "Press 'ESC' to exit, 'Q' to quit without saving", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_roi_rect(self, roi_name: str) -> Optional[Tuple[int, int, int, int]]:
        """
        获取ROI的矩形边界框
        返回: (x, y, width, height) 或 None
        """
        if roi_name not in self.roi_points:
            return None
            
        roi = self.roi_points[roi_name]
        
        # 检查所有点是否都已标定
        if not all(point is not None for point in roi.values()):
            return None
        
        # 获取所有点的x和y坐标
        points = [roi['top_left'], roi['top_right'], roi['bottom_right'], roi['bottom_left']]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # 计算边界框
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def get_roi_polygon(self, roi_name: str) -> Optional[np.ndarray]:
        """
        获取ROI的多边形点集
        返回: numpy array of points 或 None
        """
        if roi_name not in self.roi_points:
            return None
            
        roi = self.roi_points[roi_name]
        
        # 检查所有点是否都已标定
        if not all(point is not None for point in roi.values()):
            return None
        
        # 返回按顺序排列的点
        return np.array([
            roi['top_left'],
            roi['top_right'],
            roi['bottom_right'],
            roi['bottom_left']
        ], np.int32)
    
    def save_calibration(self, filename: str = "basketball_roi_calibration.json") -> bool:
        """保存标定数据到JSON文件"""
        if not self.is_calibrated:
            print("标定未完成，无法保存")
            return False
        
        calibration_data = {
            'roi_points': {
                roi_name: {
                    corner: list(point) if point is not None else None
                    for corner, point in roi_data.items()
                }
                for roi_name, roi_data in self.roi_points.items()
            },
            'roi_rects': {
                roi_name: self.get_roi_rect(roi_name)
                for roi_name in self.roi_points.keys()
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
    
    def load_calibration(self, filename: str = "basketball_roi_calibration.json") -> bool:
        """从JSON文件加载标定数据"""
        try:
            with open(filename, 'r') as f:
                calibration_data = json.load(f)
            
            # 恢复ROI点
            for roi_name, roi_data in calibration_data['roi_points'].items():
                for corner, point in roi_data.items():
                    if point is not None:
                        self.roi_points[roi_name][corner] = tuple(point)
                    else:
                        self.roi_points[roi_name][corner] = None
            
            self.is_calibrated = calibration_data['is_calibrated']
            
            if self.is_calibrated:
                self.current_state = CalibrationState.COMPLETED
            
            print(f"标定数据已从 {filename} 加载")
            return True
            
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False
        except Exception as e:
            print(f"加载失败: {e}")
            return False
    
    def reset_calibration(self):
        """重置标定"""
        for roi_name in self.roi_points:
            for corner in self.roi_points[roi_name]:
                self.roi_points[roi_name][corner] = None
        
        self.current_state = CalibrationState.WAITING
        self.is_calibrated = False
        print("\n标定已重置")
    
    def calibrate_from_image(self, image: np.ndarray) -> bool:
        """
        使用给定图像进行标定（交互式）
        Args:
            image: OpenCV格式的图像
        Returns:
            bool: 标定是否成功
        """
        self.current_image = image.copy()
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n开始交互式标定...")
        print("按 'C' 开始标定")
        print("按 'S' 保存标定结果")
        print("按 'R' 重置标定")
        print("按 'ESC' 完成标定")
        
        while True:
            self.draw_calibration_ui()
            if self.display_image is not None:
                cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('c') or key == ord('C'):
                if self.current_state == CalibrationState.WAITING:
                    print("\n开始标定，请依次点击12个标定点")
                    self.current_state = CalibrationState.CALIBRATING_BACKBOARD_TL
            elif key == ord('s') or key == ord('S'):
                if self.current_state == CalibrationState.COMPLETED:
                    self.save_calibration()
            elif key == ord('r') or key == ord('R'):
                self.reset_calibration()
        
        cv2.destroyWindow(self.window_name)
        return self.is_calibrated

def main():
    """主函数 - 使用ZED相机进行实时标定"""
    print("="*60)
    print("篮球ROI标定系统")
    print("="*60)
    
    # 初始化相机
    print("\n初始化ZED相机...")
    zed = sl.Camera()
    
    # 设置初始化参数
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 标定时不需要深度
    init_params.camera_fps = 30
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
    calibrator = BasketballROICalibrator()
    
    # 创建窗口
    window_name = calibrator.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, camera_res.width, camera_res.height)
    cv2.setMouseCallback(window_name, calibrator.mouse_callback)
    
    print("\n标定说明:")
    print("-"*40)
    print("1. 按 'C' 键开始标定")
    print("2. 依次点击12个标定点:")
    print("   篮板ROI: 左上 -> 左下 -> 右下 -> 右上")
    print("   篮筐上方ROI: 左上 -> 左下 -> 右下 -> 右上")
    print("   篮网ROI: 左上 -> 左下 -> 右下 -> 右上")
    print("3. 标定完成后按 'S' 保存")
    print("4. 按 'R' 重新标定")
    print("5. 按 'L' 加载已有标定文件")
    print("6. 按 'ESC' 退出")
    print("-"*40)
    
    try:
        while True:
            # 获取图像
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 获取左目图像
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU)
                
                # 转换为OpenCV格式
                image_ocv = image_zed.get_data()
                if image_ocv is not None:
                    # BGR转换
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
                    print("\n开始标定，请依次点击12个标定点")
                    calibrator.current_state = CalibrationState.CALIBRATING_BACKBOARD_TL
                    
            elif key == ord('s') or key == ord('S'):
                if calibrator.current_state == CalibrationState.COMPLETED:
                    calibrator.save_calibration()
                    
            elif key == ord('l') or key == ord('L'):
                calibrator.load_calibration()
                    
            elif key == ord('r') or key == ord('R'):
                calibrator.reset_calibration()
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        
    finally:
        # 清理资源
        zed.close()
        cv2.destroyAllWindows()
        print("程序退出")

# 供其他模块调用的独立函数
def calibrate_basketball_roi_from_file(image_path: str, save_path: str = "basketball_roi_calibration.json") -> Optional[BasketballROICalibrator]:
    """
    从图像文件进行标定
    Args:
        image_path: 图像文件路径
        save_path: 保存标定结果的路径
    Returns:
        标定器实例或None
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    calibrator = BasketballROICalibrator()
    success = calibrator.calibrate_from_image(image)
    
    if success:
        calibrator.save_calibration(save_path)
        return calibrator
    return None

def load_basketball_roi_calibration(calibration_path: str = "basketball_roi_calibration.json") -> Optional[BasketballROICalibrator]:
    """
    加载已有的标定文件
    Args:
        calibration_path: 标定文件路径
    Returns:
        标定器实例或None
    """
    calibrator = BasketballROICalibrator()
    if calibrator.load_calibration(calibration_path):
        return calibrator
    return None

if __name__ == "__main__":
    main()