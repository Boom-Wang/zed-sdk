#!/usr/bin/env python3
import numpy as np
from collections import deque
import argparse
from itertools import islice
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Thread, Lock
from queue import Queue, Empty
from time import sleep, time
from enum import Enum
import json
import os
from dataclasses import dataclass
from typing import Optional, List
import traceback
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=2)
exit_signal = False
inference_fps = 0.0

# Exam related parameters
class ExamState(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    FINISHED = "Finished"

class Gender(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"

# Volleyball exam parameters
EXAM_DURATION = 41
INITIAL_HEIGHT = 1.2  # Initial state height threshold (meters)

# 帧缓冲配置
BUFFER_SIZE = 7  # 缓冲帧数
PEAK_THRESHOLD = 2.0  # 像素变化阈值
PROCESS_FRAMES_BEFORE = 3  # 峰值前处理帧数
PROCESS_FRAMES_AFTER = 3  # 峰值后处理帧数

class FPSCounter:
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
        self.fps = 0.0
    
    def update(self):
        current_time = time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) > 1:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / time_diff
        return self.fps

# 增强的球状态机
class BallState(Enum):
    INITIAL = "Initial"
    RISING = "Rising"
    PEAK_DETECTED = "Peak_Detected"  
    PROCESSING = "Processing"  
    FALLING = "Falling"

# 改进的帧数据结构 - 正确保存检测结果
@dataclass
class FrameData:
    timestamp: float
    bbox: Optional[np.ndarray]  # 2D边界框
    y_bottom: float  # 底部Y坐标
    detection_list: Optional[List] = None  # 完整的YOLO检测结果列表
    processed: bool = False  # 是否已处理深度
    frame_image: Optional[np.ndarray] = None  # 可选：保存帧图像用于调试

class PrecisePerspectiveSystem:
    """透视参考系统 - 用于判断球的高度"""
    def __init__(self):
        self.camera_height = 1.5
        self.bar_height = 2.35
        self.bar_distances = {}
        self.bar_2d_points = {}
        self.distance_to_y_pixel = {}
        self.is_calibrated = False
        
    def load_calibration(self, filename="perspective_calibration.json"):
        """加载标定数据"""
        if not os.path.exists(filename):
            logger.error(f"标定文件不存在: {filename}")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_height = data['camera_height']
            self.bar_height = data['bar_height']
            self.bar_distances = data['bar_distances']
            self.bar_2d_points = {k: tuple(v) if v else None for k, v in data['bar_2d_points'].items()}
            self.distance_to_y_pixel = {float(k): v for k, v in data['distance_to_y_pixel'].items()}
            self.is_calibrated = data['is_calibrated']
            
            logger.info(f"成功加载标定数据: {filename}")
            logger.info(f"标定点数量: {len(self.bar_2d_points)}")
            logger.info(f"距离映射点数: {len(self.distance_to_y_pixel)}")
            return True
        except Exception as e:
            logger.error(f"加载标定数据失败: {e}")
            return False
    
    def check_ball_height_by_perspective(self, ball_depth, ball_bottom_y):
        """使用透视方法判断球是否超过标准高度"""
        if not self.is_calibrated or not self.distance_to_y_pixel:
            return None, 0, None
        
        try:
            # 四舍五入到最近的0.1m
            rounded_depth = round(ball_depth, 1)
            
            # 查找参考Y坐标
            reference_y = self.distance_to_y_pixel.get(rounded_depth)
            
            if reference_y is None:
                # 如果没有精确匹配，找最近的
                closest_depth = min(self.distance_to_y_pixel.keys(), 
                                   key=lambda x: abs(x - rounded_depth))
                reference_y = self.distance_to_y_pixel[closest_depth]
            
            # 判断：当球底部Y坐标 <= 参考Y坐标时，超过高度
            is_above = ball_bottom_y <= reference_y
            confidence = self._calculate_confidence(ball_depth, ball_bottom_y, reference_y)
            
            return is_above, confidence, reference_y
        except Exception as e:
            logger.error(f"高度判断出错: {e}")
            return None, 0, None
    
    def _calculate_confidence(self, depth, ball_y, ref_y):
        """计算判断置信度"""
        confidence = 0.5
        
        # 深度合理性
        if 3.45 <= depth <= 7.0:
            confidence += 0.3
        elif depth < 3.0 or depth >= 8.0:
            confidence += 0.2
        
        # Y坐标差异显著性
        y_diff = abs(ball_y - ref_y)
        if y_diff > 10:  # 显著像素差异
            confidence += 0.2
        elif y_diff > 5:
            confidence += 0.1
        return min(confidence, 1.0)
    
    def visualize_reference(self, image, ball_depth=None, ball_y=None, ref_y=None):
        """在图像上绘制参考线"""
        if not self.is_calibrated:
            return image
        
        try:
            # 绘制三根颜色不同的横杆
            colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)] 
            bar_names = ['A', 'B', 'C']
            
            # 绘制水平杆
            for i, bar in enumerate(bar_names):
                left = self.bar_2d_points.get(f'{bar}_left')
                right = self.bar_2d_points.get(f'{bar}_right')
                if left and right:
                    cv2.line(image, left, right, colors[i], 2)
                    # 添加标签
                    mid_x = (left[0] + right[0]) // 2
                    mid_y = (left[1] + right[1]) // 2
                    cv2.putText(image, f"{bar}({self.bar_distances[bar]}m)", 
                               (mid_x - 30, mid_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
            
            # 绘制连接线（左侧）
            if self.bar_2d_points.get('A_left') and self.bar_2d_points.get('B_left'):
                cv2.line(image, self.bar_2d_points['A_left'], self.bar_2d_points['B_left'], 
                        (200, 200, 200), 1)
            if self.bar_2d_points.get('B_left') and self.bar_2d_points.get('C_left'):
                cv2.line(image, self.bar_2d_points['B_left'], self.bar_2d_points['C_left'], 
                        (200, 200, 200), 1)
            
            # 绘制垂直连接（右侧）
            if self.bar_2d_points.get('A_right') and self.bar_2d_points.get('B_right'):
                cv2.line(image, self.bar_2d_points['A_right'], self.bar_2d_points['B_right'], 
                        (200, 200, 200), 1)
            if self.bar_2d_points.get('B_right') and self.bar_2d_points.get('C_right'):
                cv2.line(image, self.bar_2d_points['B_right'], self.bar_2d_points['C_right'], 
                        (200, 200, 200), 1)
            
            # 如果球信息存在，绘制球的深度和参考线
            if ball_depth is not None and ref_y is not None:
                cv2.line(image, (50, int(ref_y)), (image.shape[1] - 50, int(ref_y)), 
                        (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Standard@{ball_depth:.1f}m", 
                           (image.shape[1] - 180, int(ref_y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception as e:
            logger.error(f"可视化参考线出错: {e}")
            
        return image

class PerspectiveVolleyballCounter:
    """排球计数器 - 改进的状态管理和错误处理"""
    def __init__(self, reference_system=None):
        self.reference_system = reference_system
        
        # 计数相关
        self.count = 0
        self.current_state = BallState.INITIAL
        
        # 帧缓冲管理
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.peak_frames_to_process = []  # 待处理的峰值帧
        self.processing_lock = Lock()
        
        # 深度跟踪（用于判断上升/下降）
        self.depth_history = deque(maxlen=10)
        self.y_history = deque(maxlen=10)
        
        # 考试状态
        self.exam_state = ExamState.IDLE
        self.exam_start_time = 0
        self.exam_remaining_time = EXAM_DURATION
        
        # 峰值检测
        self.just_reached_peak = False
        self.peak_frame_index = -1
        self.peak_processing_complete = False
        self.processing_timeout = 0  # 添加处理超时计数器
        
        # 当前状态
        self.current_depth = 0
        self.current_is_above = False
        self.current_confidence = 0
        
        # 深度处理结果缓存
        self.depth_results_cache = {}
        
        # 错误计数和恢复
        self.error_count = 0
        self.max_errors = 5
        
    def add_frame_to_buffer(self, frame_data: FrameData):
        """添加帧到缓冲区 - 线程安全"""
        with self.processing_lock:
            self.frame_buffer.append(frame_data)
    
    def detect_peak(self) -> bool:
        """检测是否到达峰值 - 增强的峰值检测"""
        if len(self.y_history) < 5:
            return False
        
        try:
            # 获取最近的Y坐标变化
            recent_y = [y[1] for y in list(self.y_history)[-5:]]
            
            # 计算一阶导数
            derivatives = []
            for i in range(1, len(recent_y)):
                derivatives.append(recent_y[i] - recent_y[i-1])
            
            # 检测由负转正（从上升到下降）
            if len(derivatives) >= 2:
                # 前面是负的（上升），后面是正的（下降）
                if derivatives[-2] < -PEAK_THRESHOLD and derivatives[-1] > PEAK_THRESHOLD:
                    return True
        except Exception as e:
            logger.error(f"峰值检测出错: {e}")
            
        return False
    
    def prepare_peak_frames(self):
        """准备峰值前后的帧用于深度处理"""
        with self.processing_lock:
            try:
                buffer_length = len(self.frame_buffer)
                
                # 找到峰值帧（最后添加的帧）
                if buffer_length > 0:
                    peak_index = buffer_length - 1

                    # 获取峰值前后的帧
                    start_idx = max(0, peak_index - PROCESS_FRAMES_BEFORE)
                    end_idx = min(buffer_length, peak_index + PROCESS_FRAMES_AFTER + 1)

                    self.peak_frames_to_process = list(islice(self.frame_buffer, start_idx, end_idx))
                    self.peak_frame_index = peak_index - start_idx  # 相对索引
                    
                    logger.info(f"准备处理 {len(self.peak_frames_to_process)} 帧（峰值前{peak_index - start_idx}帧，峰值后{end_idx - peak_index - 1}帧）")
                    
                    return True
            except Exception as e:
                logger.error(f"准备峰值帧出错: {e}")
                
        return False
        
    def update_lightweight(self, bbox, y_bottom, detection_list=None):
        """轻量级更新（仅跟踪Y坐标变化）- 改进版"""
        if self.exam_state != ExamState.RUNNING:
            return
        
        try:
            current_time = time()
            # 记录Y坐标历史
            self.y_history.append((current_time, y_bottom, None))
            
            # 创建帧数据并添加到缓冲 - 保存完整的检测列表
            frame_data = FrameData(
                timestamp=current_time,
                bbox=bbox,
                y_bottom=y_bottom,
                detection_list=detection_list,  # 保存完整的检测列表
                processed=False
            )
            self.add_frame_to_buffer(frame_data)
            
            # 状态机更新
            self._update_state_lightweight(current_time)
        except Exception as e:
            logger.error(f"轻量级更新出错: {e}")
            self.error_count += 1
    
    def _update_state_lightweight(self, current_time):
        """轻量级状态更新（不进行深度计算）- 改进的状态机管理"""
        if len(self.y_history) < 3:
            return
        
        try:
            # 分析Y坐标变化趋势
            recent_y = [y[1] for y in list(self.y_history)[-5:]]
            
            if len(recent_y) >= 2:
                y_changes = []
                for i in range(1, len(recent_y)):
                    y_changes.append(recent_y[i] - recent_y[i-1])
                
                avg_change = sum(y_changes) / len(y_changes) if y_changes else 0
                
                # 状态转换 - 改进的状态机逻辑
                if self.current_state == BallState.INITIAL:
                    if avg_change < -PEAK_THRESHOLD:  # 开始上升
                        self._transition_to(BallState.RISING)
                        self.just_reached_peak = False
                        
                elif self.current_state == BallState.RISING:
                    if self.detect_peak():  # 检测到峰值
                        self._transition_to(BallState.PEAK_DETECTED)
                        self.just_reached_peak = True
                        self.peak_processing_complete = False
                        self.processing_timeout = 0
                        # 准备峰值帧用于处理
                        if self.prepare_peak_frames():
                            self._transition_to(BallState.PROCESSING)
                        
                elif self.current_state == BallState.PROCESSING:
                    # 增加超时检查
                    self.processing_timeout += 1
                    if self.processing_timeout > 30:  # 30帧超时
                        logger.warning("处理超时，强制转换到FALLING状态")
                        self._transition_to(BallState.FALLING)
                        self.peak_frames_to_process.clear()
                        self.peak_processing_complete = True
                        self.processing_timeout = 0
                    elif self.peak_processing_complete:
                        self._transition_to(BallState.FALLING)
                        self.processing_timeout = 0
                        
                elif self.current_state == BallState.FALLING:
                    # 清理状态标志
                    self.just_reached_peak = False
                    if avg_change < -PEAK_THRESHOLD:  # 再次上升
                        self._transition_to(BallState.RISING)
                        self.just_reached_peak = False
        except Exception as e:
            logger.error(f"状态更新出错: {e}")
            self.error_count += 1
            # 错误恢复：重置到初始状态
            if self.error_count > self.max_errors:
                logger.warning("错误次数过多，重置状态机")
                self._reset_state_machine()
        
    def process_peak_frames_depth(self, zed, objects_container, runtime_params):
        """处理峰值帧的深度信息 - 改进的错误处理"""
        if not self.peak_frames_to_process:
            return
        
        try:
            logger.info(f"开始处理 {len(self.peak_frames_to_process)} 个峰值帧的深度信息...")
            
            max_height_frame = None
            max_height_y = float('inf')

            # 处理每个缓冲帧
            for i, frame in enumerate(self.peak_frames_to_process):
                if frame.detection_list and not frame.processed:
                    try:
                        # 注入该帧的检测结果到ZED
                        zed.ingest_custom_box_objects(frame.detection_list)
                        # 获取跟踪的对象（包含深度信息）
                        zed.retrieve_objects(objects_container, runtime_params)
                        
                        # 处理对象深度
                        for obj in objects_container.object_list:
                            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                                ball_depth = obj.position[2]
                                
                                # 使用透视系统判断高度
                                if self.reference_system and self.reference_system.is_calibrated:
                                    is_above, confidence, ref_y = self.reference_system.check_ball_height_by_perspective(
                                        ball_depth, frame.y_bottom
                                    )
                                    
                                    # 记录深度结果
                                    self.depth_results_cache[frame.timestamp] = {
                                        'depth': ball_depth,
                                        'is_above': is_above,
                                        'confidence': confidence,
                                        'ref_y': ref_y
                                    }
                                    
                                    # 更新最高点
                                    if frame.y_bottom < max_height_y:
                                        max_height_y = frame.y_bottom
                                        max_height_depth = ball_depth
                                        max_height_frame = frame
                                    
                        frame.processed = True
                        
                    except Exception as e:
                        logger.error(f"处理帧深度时出错: {e}")
            
            # 判断最高点是否合格
            if max_height_frame and max_height_frame.timestamp in self.depth_results_cache:
                result = self.depth_results_cache[max_height_frame.timestamp]
                if result['confidence'] > 0.7 and result['is_above']:
                    self.count += 1
                    logger.info(f"✔ 排球计数 +1，总得分: {self.count}")
                    logger.info(f"  最高点深度: {result['depth']:.2f}m")
                    logger.info(f"  最高点Y坐标: {max_height_y:.1f}px")
                    logger.info(f"  置信度: {result['confidence']:.2f}")
                else:
                    logger.info("✗ 高度不足！")
                    logger.info(f"  最高点深度: {result['depth']:.2f}m")
                    logger.info(f"  最高点Y坐标: {max_height_y:.1f}px")
                    logger.info(f"  置信度: {result['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"处理峰值帧深度时出错: {e}")
        finally:
            # 清理
            self.peak_frames_to_process.clear()
            self.peak_processing_complete = True
    
    def update(self, ball_position, ball_bottom_y):
        """完整更新（包含深度信息）- 仅在需要时调用"""
        if self.exam_state != ExamState.RUNNING:
            return
        
        try:
            current_time = time()
            self.current_depth = round(ball_position[2], 1)  # Z轴是深度

            # 使用参考系统判断球的高度
            if self.reference_system and self.reference_system.is_calibrated:
                is_above, confidence, ref_y = self.reference_system.check_ball_height_by_perspective(
                    self.current_depth, ball_bottom_y
                )
                self.current_is_above = is_above if is_above is not None else False
                self.current_confidence = confidence
                
                # 记录历史
                self.depth_history.append((current_time, self.current_depth))
                self.y_history.append((current_time, ball_bottom_y, ref_y))
        except Exception as e:
            logger.error(f"完整更新出错: {e}")
    
    def _transition_to(self, new_state):
        """状态转换 - 带日志"""
        logger.info(f"状态转换: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state
    
    def _reset_state_machine(self):
        """重置状态机到初始状态"""
        self.current_state = BallState.INITIAL
        self.just_reached_peak = False
        self.peak_processing_complete = False
        self.processing_timeout = 0
        self.peak_frames_to_process.clear()
        self.error_count = 0
        logger.info("状态机已重置")
    
    def start_exam(self):
        """开始考试"""
        self.exam_state = ExamState.RUNNING
        self.exam_start_time = time()
        self.count = 0
        self.depth_history.clear()
        self.y_history.clear()
        self.frame_buffer.clear()
        self.peak_frames_to_process.clear()
        self.depth_results_cache.clear()
        self._reset_state_machine()
        logger.info(f"开始考试！时长: {EXAM_DURATION} 秒")

    def update_exam_time(self):
        """更新考试时间"""
        if self.exam_state == ExamState.RUNNING:
            elapsed = time() - self.exam_start_time
            self.exam_remaining_time = max(0, EXAM_DURATION - elapsed)
            
            if self.exam_remaining_time <= 0:
                self.finish_exam()
    
    def finish_exam(self):
        """结束考试"""
        self.exam_state = ExamState.FINISHED
        logger.info(f"考试结束！最终得分: {self.count} 个")
    
    def get_status_text(self):
        """获取状态文本"""
        status_lines = []
        status_lines.append(f"状态: {self.exam_state.value}")
        
        if self.exam_state == ExamState.RUNNING:
            status_lines.append(f"倒计时: {int(self.exam_remaining_time)} 秒")
            status_lines.append(f"排球计数: {self.count} 个")
            status_lines.append(f"当前状态: {self.current_state.value}")
            
            if self.current_state == BallState.PROCESSING:
                status_lines.append("正在处理峰值帧...")
            else:
                status_lines.append(f"缓冲区: {len(self.frame_buffer)}/{BUFFER_SIZE} 帧")

            if self.reference_system and self.reference_system.is_calibrated and self.current_depth > 0:
                status_lines.append(f"深度: {self.current_depth:.2f}m")
                status_lines.append(f"高度评估: {'合格' if self.current_is_above else '不合格'} (置信度: {self.current_confidence:.2f})")

        elif self.exam_state == ExamState.FINISHED:
            status_lines.append(f"最终得分: {self.count} 个")

        return status_lines

# 全局计数器实例
volleyball_counter = None
reference_system = None

def xywh2abcd(xywh, im_shape):
    """转换边界框格式"""
    output = np.zeros((4, 2))
    x_min = (xywh[0] - 0.5*xywh[2])
    x_max = (xywh[0] + 0.5*xywh[2])
    y_min = (xywh[1] - 0.5*xywh[3])
    y_max = (xywh[1] + 0.5*xywh[3])
    output[0][0] = x_min
    output[0][1] = y_min
    output[1][0] = x_max
    output[1][1] = y_min
    output[2][0] = x_max
    output[2][1] = y_max
    output[3][0] = x_min
    output[3][1] = y_max
    return output

def detections_to_custom_box(detections, im0, model):
    """将YOLO检测结果转换为ZED格式"""
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        
        if label == "volleyball":
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
            obj.label = cls_id
            obj.probability = float(det.conf[0])
            obj.is_grounded = False
            obj.unique_object_id = sl.generate_unique_id()
            output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.25, iou_thres=0.55):
    """优化的YOLO推理线程"""
    global exit_signal, inference_fps
    
    # 检查模型文件是否存在
    if not os.path.exists(weights):
        logger.error(f"模型文件未找到: {weights}")
        logger.error("请确保模型文件放置在正确位置")
        exit_signal = True
        return

    try:
        model = YOLO(weights)
        logger.info(f"成功加载模型: {weights}")
    except Exception as e:
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error(f"加载模型时网络错误: {e}")
        else:
            logger.error(f"加载模型出错: {e}")
        exit_signal = True
        return
    
    inference_fps_counter = FPSCounter(window_size=30)
    
    while not exit_signal:
        try:
            image_data = image_queue.get(timeout=0.1)
            if image_data.shape[2] == 4:  # RGBA
                img_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
            else:  # 已经是RGB
                img_rgb = image_data
            results = model.predict(img_rgb, save=False, imgsz=img_size, verbose=False, conf=conf_thres, iou=iou_thres)[0]
            det_boxes = results.cpu().numpy().boxes
            detections = detections_to_custom_box(det_boxes, image_data, model)
            inference_fps = inference_fps_counter.update()
            
            try:
                if detection_queue.full():
                    detection_queue.get_nowait()
                detection_queue.put(detections)
            except:
                pass
                
        except Empty:
            continue
        except Exception as e:
            logger.error(f"推理错误: {e}")

def render_2D_perspective(image, image_scale, objects, process_depth=False):
    """基于透视的2D渲染，带条件深度处理"""

    # 处理当前检测到的对象
    ball_depth = None
    ball_y = None
    ref_y = None
    
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # 获取边界框
            bbox = obj.bounding_box_2d
            
            # 计算球底部中心Y坐标
            bottom_center_x = (bbox[2][0] + bbox[3][0]) / 2
            bottom_center_y = (bbox[2][1] + bbox[3][1]) / 2  # 底部Y坐标
            
            # 缩放到显示坐标
            display_bottom_x = int(bottom_center_x * image_scale[0])
            display_bottom_y = int(bottom_center_y * image_scale[1])
            
            # 根据处理模式决定更新方式
            if process_depth and volleyball_counter:
                # 完整更新（包含深度）
                volleyball_counter.update(obj.position, bottom_center_y)
                ball_depth = obj.position[2]
                ball_y = bottom_center_y
                
                # 获取参考Y坐标
                if reference_system and reference_system.is_calibrated:
                    _, _, ref_y_temp = reference_system.check_ball_height_by_perspective(
                        ball_depth, bottom_center_y
                    )
                    ref_y = ref_y_temp
            
            # 绘制边界框 - 根据状态使用不同颜色
            if volleyball_counter:
                if volleyball_counter.current_state == BallState.PROCESSING:
                    color = (255, 0, 255)  # 紫色表示处理中
                    thickness = 3
                elif volleyball_counter.just_reached_peak:
                    color = (0, 255, 0)  # 绿色表示峰值
                    thickness = 3
                else:
                    color = (0, 200, 0)  # 普通绿色
                    thickness = 2
            else:
                color = (0, 200, 0)
                thickness = 2
            
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         color, thickness)
            
            # 绘制底部中心点
            cv2.circle(image, (display_bottom_x, display_bottom_y), 5, (255, 0, 0), -1)
            
            # 根据模式显示信息
            if process_depth:
                distance = np.linalg.norm(obj.position) if obj.position[0] != 0 else 0
                label_text = f"D={distance:.2f}m"
            else:
                label_text = f"Y={bottom_center_y:.1f}px"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 绘制参考线和判断结果
    if reference_system and reference_system.is_calibrated and ball_depth is not None and ball_y is not None:
        reference_system.visualize_reference(image, ball_depth, ball_y, ref_y)
    
    # 显示考试状态信息
    if volleyball_counter:
        volleyball_counter.update_exam_time()
        
        # 绘制半透明背景以提高文字可见度
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # 显示状态文本
        status_lines = volleyball_counter.get_status_text()
        y_offset = 30
        for line in status_lines:
            # 根据内容选择颜色
            if "合格" in line or "成功" in line:
                color = (0, 255, 255)  # 黄色表示成功
            elif "不合格" in line:
                color = (0, 0, 255)  # 红色表示失败
            elif "处理" in line:
                color = (255, 0, 255)  # 紫色表示处理中
            else:
                color = (255, 255, 255)  # 白色表示普通文本
            
            cv2.putText(image, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # 在底部显示FPS信息
        cv2.putText(image, f"Camera FPS: {camera_fps:.1f} | Inference FPS: {inference_fps:.1f}",
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# 全局FPS变量
camera_fps = 0.0

def main():
    global exit_signal, volleyball_counter, camera_fps, reference_system
    
    # 初始化前检查模型文件
    if not os.path.exists(opt.weights):
        logger.error(f"模型文件未找到: {opt.weights}")
        logger.error("请使用 --weights 参数指定正确的模型路径")
        return
    
    # 创建透视参考系统
    reference_system = PrecisePerspectiveSystem()
    
    # 加载标定数据 - 失败则退出
    if not reference_system.load_calibration():
        logger.error("标定数据未找到！")
        logger.error("请先运行标定脚本生成 perspective_calibration.json")
        return
    
    # 创建计数器
    volleyball_counter = PerspectiveVolleyballCounter(reference_system)
    
    # 启动推理线程
    capture_thread = Thread(target=torch_thread, kwargs={
        'weights': opt.weights, 
        'img_size': opt.img_size, 
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres
    })
    capture_thread.start()
    
    logger.info("初始化相机...")
    zed = sl.Camera()
    
    # 初始化参数 - 使用更轻量的深度模式
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # 轻量级深度模式
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10
    init_params.depth_minimum_distance = 2.5
    init_params.depth_stabilization = 10
    init_params.sdk_verbose = False
    init_params.enable_image_enhancement = True  # 启用图像增强

    
    # 处理输入源
    if opt.ip is not None:
        logger.info(f"连接到远程流: {opt.ip}")
        try:
            ip_address, port = opt.ip.split(':')
            port = int(port)
            init_params.set_from_stream(ip_address, port)
        except ValueError:
            logger.error("无效的IP格式。请使用格式: IP:PORT")
            exit()
    elif opt.svo is not None:
        input_type = sl.InputType()
        input_type.set_from_svo_file(opt.svo)
        init_params.input = input_type
        init_params.svo_real_time_mode = False
    
    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error(f"打开相机失败: {repr(status)}")
        exit()
    
    image_left_tmp = sl.Mat()
    logger.info("相机初始化成功")
    
    # 启用位置跟踪 - 设置为静态相机
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    positional_tracking_parameters.set_as_static = True  # 设置为静态相机
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # 配置对象检测
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    obj_param.max_range = 10
    
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        logger.error(f"启用对象检测失败: {repr(err)}。退出程序。")
        zed.close()
        exit()
    
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # 显示设置
    camera_info = zed.get_camera_information()
    camera_res = camera_info.camera_configuration.resolution
    
    # 使用较低的显示分辨率以提高性能
    display_resolution = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    
    # 创建显示图像
    image_left_ocv = np.zeros((display_resolution.height, display_resolution.width, 4), np.uint8)
    # FPS计数器
    fps_counter = FPSCounter(window_size=30)
    
    print("\n排球垫球计数系统")
    print("操作说明:")
    print("- 按 'Q' 开始/结束考试")
    print("- 按 'ESC' 退出程序")
    print("\n标准杆高度: 2.35m")
    print(f"缓冲帧数: {BUFFER_SIZE}")
    print(f"峰值检测阈值: {PEAK_THRESHOLD}px\n")
    
    # 创建窗口
    cv2.namedWindow("Volleyball Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Volleyball Counter", display_resolution.width, display_resolution.height)

    # 标志变量
    detections = []
    last_detection = None
    
    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 更新捕获FPS
                camera_fps = fps_counter.update()

                # 获取图像用于推理
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_data = image_left_tmp.get_data()
                
                # 复制图像数据到队列
                if image_data is not None and image_data.size > 0:
                    np.copyto(image_left_ocv, image_data)
                    try:
                        if image_queue.full():
                            image_queue.get_nowait()
                        image_queue.put(image_left_ocv)
                    except:
                        pass
                
                # 获取检测结果
                try:
                    detections = detection_queue.get_nowait()
                    if detections:
                        last_detection = detections
                except Empty:
                    detections = last_detection  # 使用上一次的检测结果
                
                # 修正的主循环处理逻辑
                process_depth = False
                
                # 始终注入当前检测结果（无论什么状态）
                if detections and len(detections) > 0:
                    zed.ingest_custom_box_objects(detections)
                    
                    # 更新轻量级跟踪
                    if volleyball_counter and volleyball_counter.exam_state == ExamState.RUNNING:
                        for det in detections:
                            bbox = det.bounding_box_2d
                            bottom_y = (bbox[2][1] + bbox[3][1]) / 2
                            
                            # 保存完整的检测列表
                            volleyball_counter.update_lightweight(bbox, bottom_y, detections)
                
                # PROCESSING状态下的深度处理
                if volleyball_counter and volleyball_counter.current_state == BallState.PROCESSING:
                    if volleyball_counter.peak_frames_to_process:
                        # 使用缓存的帧数据进行深度处理
                        volleyball_counter.process_peak_frames_depth(zed, objects, obj_runtime_param)
                        process_depth = True
                
                # 获取跟踪的对象（用于显示）
                if detections and len(detections) > 0:
                    zed.retrieve_objects(objects, obj_runtime_param)

                # 渲染
                render_2D_perspective(image_left_ocv, image_scale, objects, process_depth)
                cv2.imshow("Volleyball Counter", image_left_ocv)

                # 键盘控制
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    exit_signal = True
                elif key == ord('q') or key == ord('Q'):
                    if volleyball_counter.exam_state == ExamState.IDLE or volleyball_counter.exam_state == ExamState.FINISHED:
                        volleyball_counter.start_exam()
                    elif volleyball_counter.exam_state == ExamState.RUNNING:
                        volleyball_counter.finish_exam()
            else:
                logger.warning("相机抓取失败")
                sleep(0.01)
                
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        traceback.print_exc()
    finally:
        exit_signal = True
        
    logger.info("清理资源...")
    capture_thread.join(timeout=2.0)
    zed.close()
    cv2.destroyAllWindows()
    logger.info("程序退出")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/volleyball.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()