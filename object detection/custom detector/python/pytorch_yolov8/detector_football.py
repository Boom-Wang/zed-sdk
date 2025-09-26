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
def xywh2abcd(xywh, im_shape):
    x_min = (xywh[0]-0.5*xywh[2])
    x_max = (xywh[0]+0.5*xywh[2])
    y_min = (xywh[1]-0.5*xywh[3])
    y_max = (xywh[1]+0.5*xywh[3])
    return np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float32)

def detections_to_custom_box(detections, im0, model, positive_class_names):
    """将YOLO检测结果转换为ZED自定义框（仅保留足球）"""
    out = []
    for det in detections:
        xywh = det.xywh[0]
        cls_id = int(det.cls[0])
        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
        if label.lower() in positive_class_names:
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
            obj.label = cls_id
            obj.probability = float(det.conf[0])
            obj.is_grounded = False
            obj.unique_object_id = sl.generate_unique_id()
            out.append(obj)
    return out

def torch_thread(weights, img_size, conf_thres, iou_thres, class_names):
    """YOLO 推理线程"""
    global exit_signal, inference_fps
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[ERR] load model: {e}")
        exit_signal = True
        return

    positive = set([n.strip().lower() for n in class_names.split(",") if n.strip()])
    fps_counter = FPSCounter(30)
    
    while not exit_signal:
        try:
            img_rgba = image_queue.get(timeout=0.1)
        except Empty:
            continue
        try:
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
            res = model.predict(img_rgb, save=False, imgsz=img_size,
                                conf=conf_thres, iou=iou_thres, verbose=False)[0]
            det_boxes = res.cpu().numpy().boxes
            dets = detections_to_custom_box(det_boxes, img_rgba, model, positive)
            inference_fps = fps_counter.update()

            if detection_queue.full():
                try: detection_queue.get_nowait()
                except: pass
            detection_queue.put(dets)
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

# ----------------------------
# 业务逻辑：简化版射门计数器
# ----------------------------
class SoccerCounter:
    def __init__(self, goal_poly_disp, start_p1_disp, start_p2_disp,
                 depth_min=3.0, depth_max=4.2,
                 collection_time=1.0):  # 收集数据的时间窗口
        
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        
        self.goal_poly = goal_poly_disp
        self.start_p1 = start_p1_disp
        self.start_p2 = start_p2_disp
        self.crossbar_y = goal_poly_disp[3][1]  # D点y坐标（横梁高度）
        
        # 参数
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.collection_time = collection_time
        
        # 相交后的数据收集
        self.intersection_start_time = None
        self.x_values = []  # 存储x坐标值
        self.depth_values = []  # 存储深度值
        self.last_x = None  # 上一个x值，用于判断增减
        
        # 调试信息
        self.debug_info = ""

    # ---------------- 状态控制 ----------------
    def start(self):
        self.exam_state = ExamState.RUNNING
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self._clear_collection_data()
        print("Exam started")

    def stop_and_reset(self):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self._clear_collection_data()
        print("Exam stopped & reset")

    def _clear_collection_data(self):
        """清空收集的数据"""
        self.intersection_start_time = None
        self.x_values = []
        self.depth_values = []
        self.last_x = None
        self.debug_info = ""

    # ---------------- 基础几何 ----------------
    def _is_right_of_startline(self, cx, cy):
        x_line = x_on_line_at_y(self.start_p1, self.start_p2, cy)
        return cx > x_line + 2.0
        
    def _crossed_startline_to_left(self, prev_cx, prev_cy, curr_cx, curr_cy):
        x_line_prev = x_on_line_at_y(self.start_p1, self.start_p2, prev_cy)
        x_line_curr = x_on_line_at_y(self.start_p1, self.start_p2, curr_cy)
        return (prev_cx > x_line_prev + 2.0) and (curr_cx <= x_line_curr + 2.0) and (curr_cx < prev_cx - 1.0)

    # ---------------- 判断逻辑 ----------------
    def _check_goal_conditions(self):
        """检查进球条件：x值30%持续减少，深度30%符合条件"""
        if len(self.x_values) < 2:
            return False
            
        # 计算x值减少的比例
        x_decreasing_count = 0
        for i in range(1, len(self.x_values)):
            if self.x_values[i] < self.x_values[i-1]:
                x_decreasing_count += 1
        x_decreasing_ratio = x_decreasing_count / max(1, len(self.x_values) - 1)
        
        # 计算深度符合条件的比例
        valid_depth_count = sum(1 for d in self.depth_values 
                               if d is not None and self.depth_min <= d <= self.depth_max)
        depth_valid_ratio = valid_depth_count / max(1, len(self.depth_values))
        
        self.debug_info = f"X_dec:{x_decreasing_ratio:.2f} Depth_valid:{depth_valid_ratio:.2f}"
        
        return x_decreasing_ratio >= 0.3 and depth_valid_ratio >= 0.3

    def _check_out_conditions(self):
        """检查出界条件"""
        if not self.x_values:
            return False, ""
            
        # 条件2: x值30%持续增大
        if len(self.x_values) >= 2:
            x_increasing_count = 0
            for i in range(1, len(self.x_values)):
                if self.x_values[i] > self.x_values[i-1]:
                    x_increasing_count += 1
            x_increasing_ratio = x_increasing_count / (len(self.x_values) - 1)
            if x_increasing_ratio >= 0.3:
                return True, f"X increasing {x_increasing_ratio:.2f}"
        
        # 条件4: 深度值30%不符合条件
        if self.depth_values:
            invalid_depth_count = sum(1 for d in self.depth_values 
                                     if d is None or d < self.depth_min or d > self.depth_max)
            invalid_ratio = invalid_depth_count / len(self.depth_values)
            if invalid_ratio >= 0.3:
                return True, f"Invalid depth {invalid_ratio:.2f}"
        
        # 条件5: 0.2秒内深度和x都增大
        current_time = time()
        if self.intersection_start_time and (current_time - self.intersection_start_time) <= 0.2:
            if len(self.x_values) >= 2 and len(self.depth_values) >= 2:
                # 检查最近的值是否都在增大
                recent_x_increasing = all(self.x_values[i] > self.x_values[i-1] 
                                         for i in range(max(1, len(self.x_values)-2), len(self.x_values)))
                recent_depth_increasing = all(self.depth_values[i] is not None and 
                                             self.depth_values[i-1] is not None and
                                             self.depth_values[i] > self.depth_values[i-1] 
                                             for i in range(max(1, len(self.depth_values)-2), len(self.depth_values)))
                if recent_x_increasing and recent_depth_increasing:
                    return True, "X&Depth both increasing in 0.2s"
        
        return False, ""

    def _finish_shot(self, is_goal=False, reason=""):
        """结束一次射门"""
        if is_goal:
            self.goal_count += 1
            print(f"⚽ GOAL! Total: {self.goal_count} | {reason}")
        else:
            print(f"❌ OUT | {reason}")
        
        self.shoot_state = ShootState.FINISHED
        self._clear_collection_data()

    # ---------------- 每帧更新 ----------------
    def update(self, cx_disp, cy_disp, depth_m, rect_disp, detected=True):
        if self.exam_state != ExamState.RUNNING:
            return None
            
        current_time = time()
        
        # 获取bbox的上边缘y坐标（用于判断是否高于横梁）
        top_y = rect_disp[0][1] if (detected and rect_disp) else None
        
        # --- 状态机逻辑 ---
        if self.shoot_state == ShootState.PREPARE:
            if detected and self.last_x is not None:
                # 检查是否越过起点线
                if self._crossed_startline_to_left(self.last_x, cy_disp, cx_disp, cy_disp):
                    self.shoot_state = ShootState.SHOOTING
                    self.shot_total += 1
                    print(f"Start shooting #{self.shot_total}")
            
            if detected:
                self.last_x = cx_disp

        elif self.shoot_state == ShootState.SHOOTING:
            # 出界条件1: 深度不符合条件直接出界
            if detected and depth_m is not None:
                if depth_m < self.depth_min or depth_m > self.depth_max:
                    self._finish_shot(False, f"Invalid depth {depth_m:.2f}m in shooting")
                    return {"event": "out", "info": f"Invalid depth {depth_m:.2f}m"}
            
            # 出界条件3: y轴小于横梁高度
            if detected and top_y is not None and top_y < self.crossbar_y:
                self._finish_shot(False, f"Above crossbar")
                return {"event": "out", "info": "Above crossbar"}
            
            # 检查是否与ROI相交
            if detected and rect_disp and rect_poly_intersect(rect_disp, self.goal_poly):
                self.shoot_state = ShootState.INTERSECTED
                self.intersection_start_time = current_time
                self.x_values = []
                self.depth_values = []
                print("Ball intersects ROI, collecting data...")

        elif self.shoot_state == ShootState.INTERSECTED:
            # 收集数据
            if detected:
                self.x_values.append(cx_disp)
                self.depth_values.append(depth_m)
                
                # 出界条件3: y轴小于横梁高度
                if top_y is not None and top_y < self.crossbar_y:
                    self._finish_shot(False, "Above crossbar")
                    return {"event": "out", "info": "Above crossbar"}
            
            # 检查是否超过收集时间窗口
            if current_time - self.intersection_start_time >= self.collection_time:
                # 时间到，进行判定
                if self._check_goal_conditions():
                    self._finish_shot(True, self.debug_info)
                    return {"event": "goal", "info": self.debug_info}
                else:
                    self._finish_shot(False, self.debug_info)
                    return {"event": "out", "info": self.debug_info}
            else:
                # 还在收集窗口内，检查出界条件
                is_out, out_reason = self._check_out_conditions()
                if is_out:
                    self._finish_shot(False, out_reason)
                    return {"event": "out", "info": out_reason}

        elif self.shoot_state == ShootState.FINISHED:
            # 等待球回到起点线右侧
            if detected and self._is_right_of_startline(cx_disp, cy_disp):
                self.shoot_state = ShootState.PREPARE
                print("Ready for next shot")
        
        return None

    def get_collection_progress(self):
        """获取数据收集进度"""
        if self.shoot_state == ShootState.INTERSECTED and self.intersection_start_time:
            elapsed = time() - self.intersection_start_time
            return min(1.0, elapsed / self.collection_time)
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
    # 深度参数
    parser.add_argument('--depth_min', type=float, default=3.0)
    parser.add_argument('--depth_max', type=float, default=4.2)
    parser.add_argument('--collection_time', type=float, default=1.0)
    # ROI 基准分辨率
    parser.add_argument('--roi_base_w', type=int, default=1280)
    parser.add_argument('--roi_base_h', type=int, default=720)
    opt = parser.parse_args()

    # 启动推理线程
    infer_thread = Thread(target=torch_thread, kwargs=dict(
        weights=opt.weights, img_size=opt.img_size,
        conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
        class_names=opt.class_names
    ))
    infer_thread.start()

    # ZED 相机初始化
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD720
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    # init_params.depth_maximum_distance = 10.0
    # init_params.depth_minimum_distance = 0.5
    # init_params.depth_stabilization = 50

    runtime_params = sl.RuntimeParameters()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed")
        return

    # 仅对象跟踪
    # obj_param = sl.ObjectDetectionParameters()
    # obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    # obj_param.enable_tracking = True
    # obj_param.enable_segmentation = False
    # zed.enable_object_detection(obj_param)

    # 分辨率与缩放
    cam_info = zed.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    sx = display_res.width / float(opt.roi_base_w)
    sy = display_res.height / float(opt.roi_base_h)
    image_scale = (display_res.width / cam_res.width, display_res.height / cam_res.height)

    # ROI和起点线配置（基于1280x720）
    A = (92, 528); B = (92, 670); C = (250, 590); D = (250, 494)
    S1 = (1023, 587); S2 = (1170, 662)
    goal_poly_disp = scale_points([A, B, C, D], sx, sy)
    start_p1_disp, start_p2_disp = scale_points([S1, S2], sx, sy)

    counter = SoccerCounter(
        goal_poly_disp, start_p1_disp, start_p2_disp,
        depth_min=opt.depth_min, depth_max=opt.depth_max,
        collection_time=opt.collection_time
    )

    # ZED 图像/对象容器
    img_left = sl.Mat()
    img_left_net = sl.Mat()
    objs = sl.Objects()
    obj_runtime = sl.ObjectDetectionRuntimeParameters()

    # FPS
    fps_cam = FPSCounter(30)

    # UI
    cv2.namedWindow("Soccer Shoot Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Soccer Shoot Counter", display_res.width, display_res.height)

    print("\nSoccer shoot counter (simplified)")
    print("Controls: [C] start | [S] stop/reset | [Q/ESC] quit")
    print("-" * 40)

    # 主循环
    try:
        while not exit_signal:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005)
                continue

            camera_fps = fps_cam.update()

            # 获取推理图像
            zed.retrieve_image(img_left_net, sl.VIEW.LEFT)
            net_rgba = img_left_net.get_data()
            if net_rgba is not None and net_rgba.size > 0:
                if image_queue.full():
                    try: image_queue.get_nowait()
                    except: pass
                image_queue.put(net_rgba.copy())

            # 获取检测结果
            dets = None
            try:
                dets = detection_queue.get_nowait()
            except Empty:
                pass

            if dets and len(dets) > 0:
                zed.ingest_custom_box_objects(dets)
            zed.retrieve_objects(objs, obj_runtime)

            # 获取显示图像
            zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            frame = img_left.get_data()
            if frame is None:
                continue
            img = frame.copy()

            # 渲染ROI和起点线
            render_overlay(img, counter)

            # 处理检测到的对象
            event_info = None
            detected = False
            cx_disp = cy_disp = depth_m = None
            rect_disp = None

            for obj in objs.object_list:
                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    detected = True
                    bbox = obj.bounding_box_2d
                    rect_disp = [(int(bbox[i][0]*image_scale[0]), 
                                  int(bbox[i][1]*image_scale[1])) for i in range(4)]
                    cx_disp = (rect_disp[0][0] + rect_disp[2][0]) * 0.5
                    cy_disp = (rect_disp[0][1] + rect_disp[2][1]) * 0.5
                    depth_m = float(obj.position[2]) if np.isfinite(obj.position[2]) else None

                    # 画框
                    color = (0, 255, 0)
                    if counter.shoot_state == ShootState.INTERSECTED:
                        color = (0, 255, 255)
                    elif counter.shoot_state == ShootState.SHOOTING:
                        color = (255, 255, 0)

                    cv2.rectangle(img, rect_disp[0], rect_disp[2], color, 2)
                    cv2.circle(img, (int(cx_disp), int(cy_disp)), 4, (255, 0, 0), -1)
                    
                    # 显示深度
                    label = f"D={depth_m:.2f}m" if depth_m is not None else "D=--"
                    cv2.putText(img, label, (rect_disp[0][0], rect_disp[0][1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    break  # 只取一个球

            # 更新计数器状态
            ev = counter.update(cx_disp, cy_disp, depth_m, rect_disp, detected)
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
            cv2.putText(img, f"CamFPS: {camera_fps:.1f} | InfFPS: {inference_fps:.1f}",
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

            cv2.imshow("Soccer Shoot Counter", img)

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