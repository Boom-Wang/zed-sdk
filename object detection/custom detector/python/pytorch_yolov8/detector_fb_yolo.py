#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import deque, defaultdict
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Thread
from queue import Queue, Empty
from time import sleep, time
from enum import Enum
import os
import math

image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=4)
exit_signal = False
inference_fps = 0.0
camera_fps = 0.0

# ----------------------------
# 考试状态
# ----------------------------
class ExamState(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    FINISHED = "Finished"

# 射门状态机
class ShootState(Enum):
    PREPARE = "Prepare"         # 起点线右侧，等待射门
    SHOOTING = "Shooting"       # 已越过起点线且向左运动
    INTERSECTED = "Intersected" # 球与门框ROI相交，收集数据中
    FINISHED = "Finished"       # 一脚结束

# FPS 统计
class FPSCounter:
    def __init__(self, window_size=30):
        self.ts = deque(maxlen=window_size)
        self.fps = 0.0
    
    def update(self):
        now = time()
        self.ts.append(now)
        if len(self.ts) > 1:
            self.fps = (len(self.ts)-1)/(self.ts[-1]-self.ts[0])
        return self.fps

# ----------------------------
# 工具函数
# ----------------------------
def torch_thread(weights, img_size, conf_thres, iou_thres, class_names, tracker_type='bytetrack'):
    """YOLO 推理线程（使用内置跟踪）"""
    global exit_signal, inference_fps
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[ERR] load model: {e}")
        exit_signal = True
        return

    positive = set([n.strip().lower() for n in class_names.split(",") if n.strip()])
    fps_counter = FPSCounter(30)
    
    # 用于存储当前跟踪的目标ID
    tracked_id = None
    
    while not exit_signal:
        try:
            img_bgr = image_queue.get(timeout=0.1)
        except Empty:
            continue
        try:
            # 使用track方法进行检测和跟踪
            # persist=True保持跟踪器状态
            # tracker参数指定跟踪算法：bytetrack.yaml 或 botsort.yaml
            results = model.track(img_bgr, 
                                 persist=True,
                                 tracker=f"{tracker_type}.yaml",  # 使用bytetrack
                                 conf=conf_thres, 
                                 iou=iou_thres,
                                 imgsz=img_size,
                                 verbose=False)
            
            # 处理跟踪结果
            tracked_objects = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # 获取ID（如果有的话）
                    ids = boxes.id
                    
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                        
                        if label.lower() in positive:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # 获取跟踪ID（如果存在）
                            track_id = None
                            if ids is not None and i < len(ids):
                                track_id = int(ids[i])
                            
                            tracked_objects.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': confidence,
                                'class': cls_id,
                                'label': label,
                                'track_id': track_id
                            })
            
            # 选择要跟踪的目标
            selected_object = None
            
            if len(tracked_objects) > 0:
                # 如果已有跟踪目标，优先选择相同ID的
                if tracked_id is not None:
                    for obj in tracked_objects:
                        if obj['track_id'] == tracked_id:
                            selected_object = obj
                            break
                
                # 如果没找到之前的目标，选择置信度最高的
                if selected_object is None:
                    selected_object = max(tracked_objects, key=lambda x: x['conf'])
                    if selected_object['track_id'] is not None:
                        tracked_id = selected_object['track_id']
                    else:
                        tracked_id = None
            else:
                tracked_id = None
            
            inference_fps = fps_counter.update()

            if detection_queue.full():
                try: detection_queue.get_nowait()
                except: pass
            detection_queue.put(selected_object)
            
        except Exception as e:
            print(f"[Inference Error] {e}")

# 计算两点式直线在给定 y 处的 x
def x_on_line_at_y(p1, p2, y):
    x1, y1 = p1; x2, y2 = p2
    if abs(y2-y1) < 1e-6:
        return max(x1, x2)
    t = (y - y1) / (y2 - y1)
    x = x1 + t*(x2-x1)
    return x

# 线段相交判定
def seg_intersect(p1, p2, q1, q2):
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    d1 = cross(p1,p2,q1); d2 = cross(p1,p2,q2)
    d3 = cross(q1,q2,p1); d4 = cross(q1,q2,p2)
    if (d1==0 and min(p1[0],p2[0])<=q1[0]<=max(p1[0],p2[0]) and min(p1[1],p2[1])<=q1[1]<=max(p1[1],p2[1])): return True
    if (d2==0 and min(p1[0],p2[0])<=q2[0]<=max(p1[0],p2[0]) and min(p1[1],p2[1])<=q2[1]<=max(p1[1],p2[1])): return True
    if (d3==0 and min(q1[0],q2[0])<=p1[0]<=max(q1[0],q2[0]) and min(q1[1],q2[1])<=p1[1]<=max(q1[1],q2[1])): return True
    if (d4==0 and min(q1[0],q2[0])<=p2[0]<=max(q1[0],q2[0]) and min(q1[1],q2[1])<=p2[1]<=max(q1[1],q2[1])): return True
    return (d1*d2<0) and (d3*d4<0)

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32), pt, False) >= 0

def rect_poly_intersect(rect, poly):
    # 任一点包含
    for r in rect:
        if point_in_poly((r[0],r[1]), poly):
            return True
    # 多边形点在矩形内
    x_min = min([r[0] for r in rect]); x_max = max([r[0] for r in rect])
    y_min = min([r[1] for r in rect]); y_max = max([r[1] for r in rect])
    for p in poly:
        if x_min<=p[0]<=x_max and y_min<=p[1]<=y_max:
            return True
    # 边与边相交
    rect_edges = [(rect[i], rect[(i+1)%4]) for i in range(4)]
    poly_edges = [(poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))]
    for e1 in rect_edges:
        for e2 in poly_edges:
            if seg_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False

def scale_points(pts, sx, sy):
    return [(int(p[0]*sx), int(p[1]*sy)) for p in pts]

def calculate_intersection_area_ratio(rect, poly):
    """计算矩形与多边形相交面积占矩形面积的比例"""
    try:
        # 转换为numpy数组
        rect_np = np.array(rect, dtype=np.float32)
        poly_np = np.array(poly, dtype=np.float32)
        
        # 计算矩形面积
        rect_area = cv2.contourArea(rect_np)
        if rect_area < 1e-6:
            return 0.0
        
        # 计算相交区域
        x_min = int(min(min([p[0] for p in rect]), min([p[0] for p in poly])))
        x_max = int(max(max([p[0] for p in rect]), max([p[0] for p in poly])))
        y_min = int(min(min([p[1] for p in rect]), min([p[1] for p in poly])))
        y_max = int(max(max([p[1] for p in rect]), max([p[1] for p in poly])))
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # 创建掩码
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # 调整坐标
        rect_shifted = [(int(p[0]-x_min), int(p[1]-y_min)) for p in rect]
        poly_shifted = [(int(p[0]-x_min), int(p[1]-y_min)) for p in poly]
        
        cv2.fillPoly(mask1, [np.array(rect_shifted, dtype=np.int32)], 255)
        cv2.fillPoly(mask2, [np.array(poly_shifted, dtype=np.int32)], 255)
        
        # 计算相交面积
        intersection = cv2.bitwise_and(mask1, mask2)
        intersection_area = np.sum(intersection == 255)
        
        return intersection_area / rect_area
    except:
        return 0.0

def fit_line_least_squares(points):
    """使用最小二乘法拟合直线，返回方向向量"""
    if len(points) < 2:
        return None
    
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if abs(denominator) < 1e-6:  # 垂直线
        return np.array([0, 1])
    
    slope = numerator / denominator
    # 返回归一化的方向向量
    direction = np.array([1, slope])
    return direction / np.linalg.norm(direction)

def calculate_angle_between_vectors(v1, v2):
    """计算两个向量之间的夹角（度）"""
    if v1 is None or v2 is None:
        return 0
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

# ----------------------------
# 业务逻辑：2D版射门计数器
# ----------------------------
class SoccerCounter:
    def __init__(self, goal_poly_disp, start_p1_disp, start_p2_disp,
                 before_time=1.0, after_time=0.5, angle_threshold=150.0,
                 x_increase_ratio=0.3, intersection_ratio_threshold=0.15,
                 shooting_timeout=3.0):
        
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        
        self.goal_poly = goal_poly_disp
        self.start_p1 = start_p1_disp
        self.start_p2 = start_p2_disp
        self.crossbar_y = goal_poly_disp[3][1]  # D点y坐标（横梁高度）
        
        # 参数
        self.before_time = before_time  # 相交前记录时间
        self.after_time = after_time    # 相交后记录时间
        self.angle_threshold = angle_threshold  # V字判定角度阈值
        self.x_increase_ratio = x_increase_ratio  # x增大比例阈值
        self.intersection_ratio_threshold = intersection_ratio_threshold  # 相交面积比例阈值
        self.shooting_timeout = shooting_timeout  # 射门超时时间
        
        # 轨迹记录
        self.trajectory_before = deque(maxlen=100)  # 相交前轨迹
        self.trajectory_after = []   # 相交后轨迹
        self.trajectory_timestamps = deque(maxlen=100)  # 时间戳
        
        # 状态记录
        self.intersection_start_time = None
        self.shooting_start_time = None
        self.last_x = None
        self.last_cx = None
        self.last_cy = None
        
        # 顶部出界检测
        self.above_crossbar_frames = 0
        
        # 调试信息
        self.debug_info = ""
        self.trajectory_display = []  # 用于显示的轨迹点

    # ---------------- 状态控制 ----------------
    def start(self):
        self.exam_state = ExamState.RUNNING
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self._clear_all_data()
        print("Exam started")

    def stop_and_reset(self):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self._clear_all_data()
        print("Exam stopped & reset")

    def _clear_all_data(self):
        """清空所有数据"""
        self.trajectory_before.clear()
        self.trajectory_after.clear()
        self.trajectory_timestamps.clear()
        self.trajectory_display.clear()
        self.intersection_start_time = None
        self.shooting_start_time = None
        self.last_x = None
        self.last_cx = None
        self.last_cy = None
        self.above_crossbar_frames = 0
        self.debug_info = ""

    def _clear_shot_data(self):
        """清空单次射门数据"""
        self.trajectory_after.clear()
        self.intersection_start_time = None
        self.above_crossbar_frames = 0
        self.debug_info = ""

    # ---------------- 基础几何 ----------------
    def _is_right_of_startline(self, cx, cy):
        x_line = x_on_line_at_y(self.start_p1, self.start_p2, cy)
        return cx > x_line + 2.0
        
    def _crossed_startline_to_left(self, prev_cx, prev_cy, curr_cx, curr_cy):
        x_line_prev = x_on_line_at_y(self.start_p1, self.start_p2, prev_cy)
        x_line_curr = x_on_line_at_y(self.start_p1, self.start_p2, curr_cy)
        return (prev_cx > x_line_prev + 2.0) and (curr_cx <= x_line_curr + 2.0) and (curr_cx < prev_cx - 1.0)

    def _record_trajectory(self, cx, cy):
        """记录轨迹点"""
        current_time = time()
        
        # 添加到轨迹
        if self.shoot_state in [ShootState.SHOOTING, ShootState.INTERSECTED]:
            if self.intersection_start_time is None:
                # 还未相交，记录到before轨迹
                self.trajectory_before.append((cx, cy))
                self.trajectory_timestamps.append(current_time)
            else:
                # 已相交，记录到after轨迹
                self.trajectory_after.append((cx, cy))
        
        # 清理过期的before轨迹点（只保留最近before_time秒的）
        while (self.trajectory_timestamps and 
               current_time - self.trajectory_timestamps[0] > self.before_time):
            self.trajectory_timestamps.popleft()
            if self.trajectory_before:
                self.trajectory_before.popleft()

    def _get_trajectory_for_display(self):
        """获取用于显示的轨迹"""
        display_trajectory = []
        
        # 获取相交前最近1秒的轨迹
        if self.intersection_start_time:
            current_time = time()
            before_points = []
            for i, t in enumerate(self.trajectory_timestamps):
                if self.intersection_start_time - t <= self.before_time:
                    if i < len(self.trajectory_before):
                        before_points.append(self.trajectory_before[i])
            display_trajectory.extend(before_points)
        else:
            display_trajectory.extend(list(self.trajectory_before)[-30:])  # 显示最近30个点
        
        # 添加相交后的轨迹
        display_trajectory.extend(self.trajectory_after)
        
        return display_trajectory

    # ---------------- 判断逻辑 ----------------
    def _check_goal_conditions(self):
        """检查进球条件"""
        # 条件1: 轨迹整体从右向左运动（基于轨迹分析）
        if self._is_consistent_left_movement():
            self.debug_info = "Consistent left movement"
            return True
        
        # 条件2: x值30%持续减少（基于after轨迹）
        if self._check_x_decreasing():
            self.debug_info = "X decreasing >= 30%"
            return True
        
        return False

    def _is_consistent_left_movement(self):
        """判断是否持续向左运动"""
        # 合并before和after轨迹进行分析
        combined_trajectory = list(self.trajectory_before)[-20:] + self.trajectory_after
        
        if len(combined_trajectory) < 5:
            return False
        
        # 使用最小二乘法拟合轨迹
        direction = fit_line_least_squares(combined_trajectory)
        
        if direction is not None:
            # 检查主要运动方向是否向左（x分量为负）
            if direction[0] < -0.5:  # 主要向左运动
                return True
        
        # 备用方法：检查x坐标整体趋势
        x_coords = [p[0] for p in combined_trajectory]
        if len(x_coords) >= 2:
            # 计算总体趋势
            overall_trend = x_coords[-1] - x_coords[0]
            if overall_trend < -20:  # 整体向左移动超过20像素
                return True
        
        return False

    def _check_x_decreasing(self):
        """检查x值是否30%持续减少"""
        if len(self.trajectory_after) < 2:
            return False
        
        x_decreasing_count = 0
        for i in range(1, len(self.trajectory_after)):
            if self.trajectory_after[i][0] < self.trajectory_after[i-1][0]:
                x_decreasing_count += 1
        
        ratio = x_decreasing_count / (len(self.trajectory_after) - 1)
        return ratio >= self.x_increase_ratio

    def _check_out_conditions(self):
        """检查出界条件"""
        # 条件1: 顶部出界（连续2帧）
        if self.above_crossbar_frames >= 2:
            return True, "Above crossbar (2 frames)"
        
        # 条件2: V字反弹检测
        v_pattern, angle = self._detect_v_pattern()
        if v_pattern:
            return True, f"V-pattern bounce (angle: {angle:.1f}°)"
        
        # 条件3: x值30%持续增大
        if self._check_x_increasing():
            return True, "X increasing >= 30%"
        
        # 条件4: 相交面积过小（已在主循环中处理）
        
        return False, ""

    def _detect_v_pattern(self):
        """检测V字型反弹模式"""
        if len(self.trajectory_after) < 5:
            return False, 0
        
        # 获取相交前的轨迹（最近0.5秒）
        before_points = []
        if self.intersection_start_time:
            for i, t in enumerate(self.trajectory_timestamps):
                if self.intersection_start_time - t <= 0.5:
                    if i < len(self.trajectory_before):
                        before_points.append(self.trajectory_before[i])
        
        if len(before_points) < 3 or len(self.trajectory_after) < 3:
            return False, 0
        
        # 使用最小二乘法拟合两段轨迹
        v1 = fit_line_least_squares(before_points)
        v2 = fit_line_least_squares(self.trajectory_after)
        
        if v1 is not None and v2 is not None:
            angle = calculate_angle_between_vectors(v1, v2)
            if angle <= self.angle_threshold and angle > 30:  # 30度到150度之间
                return True, angle
        
        return False, 0

    def _check_x_increasing(self):
        """检查x值是否30%持续增大"""
        if len(self.trajectory_after) < 2:
            return False
        
        x_increasing_count = 0
        for i in range(1, len(self.trajectory_after)):
            if self.trajectory_after[i][0] > self.trajectory_after[i-1][0]:
                x_increasing_count += 1
        
        ratio = x_increasing_count / (len(self.trajectory_after) - 1)
        return ratio >= self.x_increase_ratio

    def _finish_shot(self, is_goal=False, reason=""):
        """结束一次射门"""
        if is_goal:
            self.goal_count += 1
            print(f"⚽ GOAL! Total: {self.goal_count} | {reason}")
        else:
            print(f"❌ OUT | {reason}")
        
        self.shoot_state = ShootState.FINISHED
        self._clear_shot_data()

    # ---------------- 每帧更新 ----------------
    def update(self, cx_disp, cy_disp, rect_disp, detected=True):
        if self.exam_state != ExamState.RUNNING:
            return None
            
        current_time = time()
        
        # 获取bbox顶部中心点y坐标
        top_center_y = None
        if detected and rect_disp:
            top_center_y = (rect_disp[0][1] + rect_disp[1][1]) / 2
        
        # --- 状态机逻辑 ---
        if self.shoot_state == ShootState.PREPARE:
            if detected and self.last_x is not None:
                # 检查是否越过起点线
                if self._crossed_startline_to_left(self.last_x, self.last_cy, cx_disp, cy_disp):
                    self.shoot_state = ShootState.SHOOTING
                    self.shooting_start_time = current_time
                    self.shot_total += 1
                    self.trajectory_before.clear()
                    self.trajectory_timestamps.clear()
                    print(f"Start shooting #{self.shot_total}")
            
            if detected:
                self.last_x = cx_disp
                self.last_cx = cx_disp
                self.last_cy = cy_disp

        elif self.shoot_state == ShootState.SHOOTING:
            # 记录轨迹
            if detected:
                self._record_trajectory(cx_disp, cy_disp)
                self.last_cx = cx_disp
                self.last_cy = cy_disp
            
            # 检查3秒超时
            if current_time - self.shooting_start_time > self.shooting_timeout:
                self._finish_shot(False, "3s timeout - no intersection")
                return {"event": "out", "info": "3s timeout"}
            
            # 检查是否与ROI相交
            if detected and rect_disp and rect_poly_intersect(rect_disp, self.goal_poly):
                self.shoot_state = ShootState.INTERSECTED
                self.intersection_start_time = current_time
                self.trajectory_after = []
                print("Ball intersects ROI, collecting data...")

        elif self.shoot_state == ShootState.INTERSECTED:
            # 记录轨迹
            if detected:
                self._record_trajectory(cx_disp, cy_disp)
                
                # 检查顶部出界（连续2帧）
                if top_center_y is not None:
                    if top_center_y < self.crossbar_y:
                        self.above_crossbar_frames += 1
                    else:
                        self.above_crossbar_frames = 0
                
                # 检查相交面积
                if rect_disp:
                    intersection_ratio = calculate_intersection_area_ratio(rect_disp, self.goal_poly)
                    if intersection_ratio < self.intersection_ratio_threshold:
                        self._finish_shot(False, f"Intersection ratio {intersection_ratio:.2%} < 15%")
                        return {"event": "out", "info": f"Side boundary (ratio: {intersection_ratio:.2%})"}
            
            # 检查是否超过收集时间窗口
            if current_time - self.intersection_start_time >= self.after_time:
                # 时间到，进行判定
                is_out, out_reason = self._check_out_conditions()
                
                if is_out:
                    self._finish_shot(False, out_reason)
                    return {"event": "out", "info": out_reason}
                elif self._check_goal_conditions():
                    self._finish_shot(True, self.debug_info)
                    return {"event": "goal", "info": self.debug_info}
                else:
                    # 默认判定为出界
                    self._finish_shot(False, "No clear goal pattern")
                    return {"event": "out", "info": "No clear goal pattern"}
            else:
                # 还在收集窗口内，实时检查出界条件
                is_out, out_reason = self._check_out_conditions()
                if is_out:
                    self._finish_shot(False, out_reason)
                    return {"event": "out", "info": out_reason}

        elif self.shoot_state == ShootState.FINISHED:
            # 等待球回到起点线右侧
            if detected and self._is_right_of_startline(cx_disp, cy_disp):
                self.shoot_state = ShootState.PREPARE
                self.shooting_start_time = None
                print("Ready for next shot")
        
        # 更新显示轨迹
        if self.shoot_state in [ShootState.SHOOTING, ShootState.INTERSECTED]:
            self.trajectory_display = self._get_trajectory_for_display()
        else:
            self.trajectory_display = []
        
        return None

    def get_collection_progress(self):
        """获取数据收集进度"""
        if self.shoot_state == ShootState.INTERSECTED and self.intersection_start_time:
            elapsed = time() - self.intersection_start_time
            return min(1.0, elapsed / self.after_time)
        return 0.0

# ----------------------------
# 渲染函数
# ----------------------------
def render_overlay(img, counter: SoccerCounter):
    # 画ROI和起点线
    cv2.polylines(img, [np.array(counter.goal_poly, dtype=np.int32)], True, (0,0,255), 2)
    cv2.line(img, counter.start_p1, counter.start_p2, (255,255,0), 2)
    
    # 画横梁高度线（调试用）
    cv2.line(img, (0, counter.crossbar_y), (img.shape[1], counter.crossbar_y), 
             (0, 255, 255), 1, cv2.LINE_AA)
    
    # 绘制轨迹
    if len(counter.trajectory_display) > 1:
        for i in range(1, len(counter.trajectory_display)):
            pt1 = (int(counter.trajectory_display[i-1][0]), int(counter.trajectory_display[i-1][1]))
            pt2 = (int(counter.trajectory_display[i][0]), int(counter.trajectory_display[i][1]))
            
            # 根据时间位置选择颜色
            if counter.intersection_start_time and i < len(counter.trajectory_display) - len(counter.trajectory_after):
                color = (0, 255, 0)  # 绿色：相交前轨迹
            else:
                color = (255, 0, 255)  # 紫色：相交后轨迹
            
            cv2.line(img, pt1, pt2, color, 2)
            cv2.circle(img, pt2, 3, color, -1)
    
    # 显示调试信息
    if counter.debug_info:
        cv2.putText(img, counter.debug_info, (20, img.shape[0]-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def main():
    global exit_signal, camera_fps, inference_fps

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/yolo11n.pt', help='YOLO model')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--class_names', type=str,
                        default='sports ball,soccer ball,football,ball',
                        help='comma separated')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                        help='Tracker type: bytetrack or botsort')
    # 新参数（不再需要深度参数）
    parser.add_argument('--before_time', type=float, default=1.0, help='Track time before intersection')
    parser.add_argument('--after_time', type=float, default=0.5, help='Track time after intersection')
    parser.add_argument('--angle_threshold', type=float, default=150.0, help='V-pattern angle threshold')
    parser.add_argument('--shooting_timeout', type=float, default=3.0, help='Shooting state timeout')
    # ROI 基准分辨率
    parser.add_argument('--roi_base_w', type=int, default=1280)
    parser.add_argument('--roi_base_h', type=int, default=720)
    opt = parser.parse_args()

    # 启动推理线程
    infer_thread = Thread(target=torch_thread, kwargs=dict(
        weights=opt.weights, img_size=opt.img_size,
        conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
        class_names=opt.class_names, tracker_type=opt.tracker
    ))
    infer_thread.start()

    # ZED 相机初始化（只作为普通相机使用）
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 完全禁用深度
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD720

    runtime_params = sl.RuntimeParameters()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    # 分辨率与缩放
    cam_info = zed.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    sx = display_res.width / float(opt.roi_base_w)
    sy = display_res.height / float(opt.roi_base_h)

    # ROI和起点线配置（基于1280x720）
    A = (92, 528); B = (92, 670); C = (250, 590); D = (250, 494)
    S1 = (1023, 587); S2 = (1170, 662)
    goal_poly_disp = scale_points([A, B, C, D], sx, sy)
    start_p1_disp, start_p2_disp = scale_points([S1, S2], sx, sy)

    counter = SoccerCounter(
        goal_poly_disp, start_p1_disp, start_p2_disp,
        before_time=opt.before_time,
        after_time=opt.after_time,
        angle_threshold=opt.angle_threshold,
        shooting_timeout=opt.shooting_timeout
    )

    # ZED 图像容器
    img_left = sl.Mat()
    img_left_net = sl.Mat()

    # FPS
    fps_cam = FPSCounter(30)

    # UI
    cv2.namedWindow("Soccer Shoot Counter 2D", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Soccer Shoot Counter 2D", display_res.width, display_res.height)

    print("\nSoccer shoot counter (2D version - ByteTrack)")
    print("Controls: [C] start | [S] stop/reset | [Q/ESC] quit")
    print("-" * 40)

    # 主循环
    try:
        while not exit_signal:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005)
                continue

            camera_fps = fps_cam.update()

            # 获取图像用于推理（原始分辨率）
            zed.retrieve_image(img_left_net, sl.VIEW.LEFT)
            net_rgba = img_left_net.get_data()
            if net_rgba is not None and net_rgba.size > 0:
                # 转换为BGR格式
                img_bgr = cv2.cvtColor(net_rgba, cv2.COLOR_BGRA2RGB)
                if image_queue.full():
                    try: image_queue.get_nowait()
                    except: pass
                image_queue.put(img_bgr.copy())

            # 获取显示图像（显示分辨率）
            zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            frame = img_left.get_data()
            if frame is None:
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            # 渲染ROI、起点线和轨迹
            render_overlay(img, counter)

            # 获取跟踪结果
            tracked_obj = None
            try:
                tracked_obj = detection_queue.get_nowait()
            except Empty:
                pass

            # 处理跟踪结果
            event_info = None
            detected = False
            cx_disp = cy_disp = None
            rect_disp = None

            if tracked_obj is not None:
                detected = True
                x1, y1, x2, y2 = tracked_obj['bbox']
                
                # 缩放到显示分辨率
                scale_x = display_res.width / cam_res.width
                scale_y = display_res.height / cam_res.height
                
                x1_disp = int(x1 * scale_x)
                y1_disp = int(y1 * scale_y)
                x2_disp = int(x2 * scale_x)
                y2_disp = int(y2 * scale_y)
                
                # 创建矩形四个角点
                rect_disp = [(x1_disp, y1_disp), (x2_disp, y1_disp), 
                             (x2_disp, y2_disp), (x1_disp, y2_disp)]
                
                # 计算中心点
                cx_disp = (x1_disp + x2_disp) * 0.5
                cy_disp = (y1_disp + y2_disp) * 0.5

                # 画框
                color = (0, 255, 0)
                if counter.shoot_state == ShootState.INTERSECTED:
                    color = (0, 255, 255)
                elif counter.shoot_state == ShootState.SHOOTING:
                    color = (255, 255, 0)

                cv2.rectangle(img, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                cv2.circle(img, (int(cx_disp), int(cy_disp)), 4, (255, 0, 0), -1)
                
                # 显示标签、置信度和跟踪ID
                label_text = f"{tracked_obj['label']} ({tracked_obj['conf']:.2f})"
                if tracked_obj['track_id'] is not None:
                    label_text += f" ID:{tracked_obj['track_id']}"
                cv2.putText(img, label_text, (x1_disp, y1_disp-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 更新计数器状态
            ev = counter.update(cx_disp, cy_disp, rect_disp, detected)
            if ev is not None:
                event_info = ev

            # 事件提示
            if event_info:
                if event_info["event"] == "goal":
                    txt, color = "GOAL +1", (0, 255, 255)
                else:
                    txt, color = "OUT", (0, 0, 255)
                cv2.putText(img, txt, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 状态信息
            cv2.putText(img, f"Status: {counter.exam_state.value} | State: {counter.shoot_state.value}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Score: {counter.goal_count} | Shots: {counter.shot_total}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"CamFPS: {camera_fps:.1f} | InfFPS: {inference_fps:.1f} | Tracker: {opt.tracker}",
                        (20, display_res.height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 显示收集进度条
            if counter.shoot_state == ShootState.INTERSECTED:
                progress = counter.get_collection_progress()
                bar_width = 200
                bar_height = 20
                bar_x = 20
                bar_y = 100
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), 
                             (0, 255, 255), -1)
                cv2.putText(img, f"Collecting: {progress*100:.0f}%", (bar_x + bar_width + 10, bar_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Soccer Shoot Counter 2D", img)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                counter.start()
            elif key == ord('s') or key == ord('S'):
                counter.stop_and_reset()

    except KeyboardInterrupt:
        pass
    finally:
        exit_signal = True
        infer_thread.join(timeout=2.0)
        zed.close()
        cv2.destroyAllWindows()
        print("\nProgram exit")

if __name__ == "__main__":
    with torch.no_grad():
        main()