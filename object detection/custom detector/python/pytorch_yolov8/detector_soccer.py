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

# ----------------------------
# 线程队列，与项目保持一致风格
# ----------------------------
image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=2)
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
    PREPARE = "Prepare"     # 起点线右侧，等待射门
    SHOOTING = "Shooting"   # 已越过起点线且向左运动
    FINISHED = "Finished"   # 一脚结束（进球/出界）

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
    """YOLO 推理线程。参考项目示例的写法保持一致（模型->预测->CustomBox->队列传递）。"""
    global exit_signal, inference_fps
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[ERR] load model: {e}")
        exit_signal = True
        return

    # 统一大小写
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

# 计算两点式直线在给定 y 处的 x（延长线也成立）
def x_on_line_at_y(p1, p2, y):
    x1, y1 = p1; x2, y2 = p2
    if abs(y2-y1) < 1e-6:
        # 水平线：返回右端点x（用于“右侧”判断）
        return max(x1, x2)
    t = (y - y1) / (y2 - y1)
    x = x1 + t*(x2-x1)
    return x

# 线段相交判定（用于矩形-多边形）
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
    # rect: [ (x1,y1),(x2,y2),(x3,y3),(x4,y4) ] (顺时针)
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
# 业务逻辑：射门计数器
# ----------------------------
class SoccerCounter:
    def __init__(self, goal_poly_disp, start_p1_disp, start_p2_disp,
                 depth_min=3.0, depth_max=4.2,
                 pre_frames=10, post_frames=5, post_min_dec=3,
                 bounce_px=4.0):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0

        self.goal_poly = goal_poly_disp  # 显示画面坐标
        self.start_p1 = start_p1_disp
        self.start_p2 = start_p2_disp
        self.crossbar_Dy = goal_poly_disp[3][1]  # D 点 y（A,B,C,D 顺时针：A左上,B左下,C右下,D右上）

        # 参数
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.post_min_dec = post_min_dec
        self.bounce_px = bounce_px

        # 历史缓存
        self.hist = deque(maxlen=200)  # 元素：(t,cx,cy,top_y,depth,bbox_rect)
        self.active_inter_idx = None   # ROI 相交帧索引
        self.await_decision = False

        # 平滑
        self.last_cx = None
        self.last_state_change = time()

    def start(self):
        self.exam_state = ExamState.RUNNING
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self.hist.clear()
        self.active_inter_idx = None
        self.await_decision = False

    def stop_and_reset(self):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self.hist.clear()
        self.active_inter_idx = None
        self.await_decision = False

    def _cx_trend(self, seq):
        """返回该序列中“向左(减小)”的帧计数"""
        cnt = 0
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1] - 1.0:  # 1px 抖动阈值
                cnt += 1
        return cnt

    def _median_depth(self, depths):
        vals = [d for d in depths if d is not None and np.isfinite(d)]
        if not vals: return None
        return float(np.median(vals))

    def _depth_ok_majority(self, depths):
        if not depths: return False
        ok = sum(1 for d in depths if d is not None and self.depth_min <= d <= self.depth_max)
        return ok >= max(1, len(depths)//2)  # 多数派

    def _is_right_of_startline(self, cx, cy):
        x_line = x_on_line_at_y(self.start_p1, self.start_p2, cy)
        return cx > x_line + 2.0  # 2px 容差

    def _crossed_startline_to_left(self, prev, curr):
        # prev/curr: (cx, cy)
        cx0, cy0 = prev; cx1, cy1 = curr
        x_line0 = x_on_line_at_y(self.start_p1, self.start_p2, cy0)
        x_line1 = x_on_line_at_y(self.start_p1, self.start_p2, cy1)
        return (cx0 > x_line0 + 2.0) and (cx1 <= x_line1 + 2.0) and (cx1 < cx0 - 1.0)

    def _rect_from_bbox_2d(self, bbox_2d, sx, sy):
        # bbox_2d 是基于原始帧坐标；显示缩放应用 sx,sy
        return [(int(bbox_2d[i][0]*sx), int(bbox_2d[i][1]*sy)) for i in range(4)]

    def _decide_goal_or_out(self):
        """在 active_inter_idx 已确定后，基于前后帧趋势做最终判定"""
        if self.active_inter_idx is None:
            return False

        L = len(self.hist)
        i0 = self.active_inter_idx

        # 提取前后窗口
        i_start = max(0, i0 - self.pre_frames)
        i_end   = min(L-1, i0 + self.post_frames)  # 包含
        window = list(self.hist)[i_start:i_end+1]

        # 相交后子序列
        post = list(self.hist)[i0:min(L, i0+self.post_frames+1)]
        cxs_post = [e[1] for e in post]
        depths_post = [e[4] for e in post]

        # 反弹快速判定：相交后前3帧出现明显回弹（连续两帧 Δx > +bounce_px）
        rebound = False
        for k in range(2, min(4, len(cxs_post))):
            dx1 = cxs_post[k-1] - cxs_post[k-2]
            dx2 = cxs_post[k]   - cxs_post[k-1]
            if dx1 > self.bounce_px and dx2 > self.bounce_px:
                rebound = True
                break

        # 一般趋势：K 帧中至少 M 帧继续向左
        left_trend_ok = self._cx_trend(cxs_post) >= self.post_min_dec
        depth_ok = self._depth_ok_majority(depths_post)

        # 横梁约束（任取相交帧与之后数帧的 top_y 均需 >= D_y）
        topys = [e[3] for e in post]
        crossbar_ok = all([ty is not None and ty >= self.crossbar_Dy for ty in topys])

        if (not rebound) and left_trend_ok and depth_ok and crossbar_ok:
            self.goal_count += 1
            result = True
        else:
            result = False

        # 一脚结束
        self.shoot_state = ShootState.FINISHED
        self.await_decision = False
        self.active_inter_idx = None
        return result

    def update(self, cx_disp, cy_disp, top_y_disp, depth_m, rect_disp):
        """每帧更新（显示坐标系下）"""
        if self.exam_state != ExamState.RUNNING:
            return None

        # 记录历史
        self.hist.append((time(), float(cx_disp), float(cy_disp),
                          float(top_y_disp) if top_y_disp is not None else None,
                          float(depth_m) if depth_m is not None else None,
                          rect_disp))

        # 状态机
        n = len(self.hist)
        if n < 2:
            return None

        cx_prev, cy_prev = self.hist[-2][1], self.hist[-2][2]
        cx_curr, cy_curr = self.hist[-1][1], self.hist[-1][2]

        if self.shoot_state == ShootState.PREPARE:
            # 起点线右侧即准备状态
            if self._is_right_of_startline(cx_curr, cy_curr):
                # 等待越线且向左
                if self._crossed_startline_to_left((cx_prev,cy_prev),(cx_curr,cy_curr)):
                    self.shoot_state = ShootState.SHOOTING
                    self.shot_total += 1
                    self.await_decision = False
            else:
                # 不在右侧，维持
                pass

        elif self.shoot_state == ShootState.SHOOTING:
            # ROI 相交检测（矩形-多边形）
            if rect_poly_intersect(rect_disp, self.goal_poly):
                # 同时满足深度与横梁初步约束才进入“候选判定”
                depth_ok_now = (depth_m is not None) and (self.depth_min <= depth_m <= self.depth_max)
                crossbar_ok_now = (top_y_disp is not None) and (top_y_disp >= self.crossbar_Dy)
                if (not self.await_decision) and depth_ok_now and crossbar_ok_now:
                    self.active_inter_idx = len(self.hist)-1
                    self.await_decision = True

            # 若已进入候选，等后续帧到齐或出现反弹特征即做最终判定
            if self.await_decision:
                # 条件：达到最少后帧数量 或 观测到快速反弹
                i0 = self.active_inter_idx
                post_len = len(self.hist) - i0
                need_decide = post_len >= self.post_frames
                if not need_decide and post_len >= 3:
                    # 提前检查反弹
                    subset = [e[1] for e in list(self.hist)[i0:len(self.hist)]]
                    if len(subset) >= 3:
                        dx1 = subset[-2]-subset[-3]
                        dx2 = subset[-1]-subset[-2]
                        if dx1 > self.bounce_px and dx2 > self.bounce_px:
                            need_decide = True
                if need_decide:
                    goal = self._decide_goal_or_out()
                    return {"event":"goal" if goal else "out"}

            # 若球回到起点线右侧（或离开画面很远）也认为一次结束（未达成候选则为出界）
            if self._is_right_of_startline(cx_curr, cy_curr):
                self.shoot_state = ShootState.FINISHED
                self.await_decision = False
                self.active_inter_idx = None
                return {"event":"out"}

        elif self.shoot_state == ShootState.FINISHED:
            # 回到准备区重新等待下一次
            if self._is_right_of_startline(cx_curr, cy_curr):
                self.shoot_state = ShootState.PREPARE

        return None

# ----------------------------
# 渲染与主循环
# ----------------------------
def render_overlay(img, counter: SoccerCounter):
    # 画 ROI、多边形与起点线
    cv2.polylines(img, [np.array(counter.goal_poly, dtype=np.int32)], True, (0,0,255), 2)
    cv2.line(img, counter.start_p1, counter.start_p2, (255,255,0), 2)
    # 延长线
    v = (counter.start_p2[0]-counter.start_p1[0], counter.start_p2[1]-counter.start_p1[1])
    L = 3000
    p_ext = (counter.start_p2[0]+int(v[0]*L/100.0), counter.start_p2[1]+int(v[1]*L/100.0))
    cv2.line(img, counter.start_p2, p_ext, (200,200,50), 1, cv2.LINE_AA)

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
    # 深度与窗口参数
    parser.add_argument('--depth_min', type=float, default=3.0)
    parser.add_argument('--depth_max', type=float, default=4.2)
    parser.add_argument('--pre_frames', type=int, default=10)
    parser.add_argument('--post_frames', type=int, default=5)
    parser.add_argument('--post_min_dec', type=int, default=3)
    parser.add_argument('--bounce_px', type=float, default=4.0)
    # ROI 基准分辨率（用于把你标注在 1280x720 上的像素点缩放到显示分辨率）
    parser.add_argument('--roi_base_w', type=int, default=1280)
    parser.add_argument('--roi_base_h', type=int, default=720)
    opt = parser.parse_args()

    # 启动推理线程（与项目相同的生产者/消费者写法）
    infer_thread = Thread(target=torch_thread, kwargs=dict(
        weights=opt.weights, img_size=opt.img_size,
        conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
        class_names=opt.class_names
    ))
    infer_thread.start()

    # ---- ZED 相机初始化（保持与项目示例一致的 API/参数风格） ----
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # 轻量深度
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10.0
    init_params.depth_minimum_distance = 0.5
    init_params.depth_stabilization = 50
    # 以上配置与项目中用法一致：打开相机->启用跟踪->自定义框注入->取对象

    runtime_params = sl.RuntimeParameters()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed"); return

    pose_param = sl.PositionalTrackingParameters()
    pose_param.set_as_static = True
    zed.enable_positional_tracking(pose_param)  # 保持与示例一致:contentReference[oaicite:3]{index=3}

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)  # 与示例一致:contentReference[oaicite:4]{index=4}

    # 分辨率与缩放
    cam_info = zed.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    sx = display_res.width / float(opt.roi_base_w)
    sy = display_res.height / float(opt.roi_base_h)
    image_scale = (display_res.width / cam_res.width, display_res.height / cam_res.height)

    # 你的 ROI/起点线（基于 1280x720 标注） -> 显示分辨率缩放
    A = (92, 528); B=(92,670); C=(250,590); D=(250,494)  # 顺时针 A,B,C,D
    S1=(1023,587); S2=(1170,662)
    goal_poly_disp = scale_points([A,B,C,D], sx, sy)
    start_p1_disp, start_p2_disp = scale_points([S1,S2], sx, sy)

    counter = SoccerCounter(
        goal_poly_disp, start_p1_disp, start_p2_disp,
        depth_min=opt.depth_min, depth_max=opt.depth_max,
        pre_frames=opt.pre_frames, post_frames=opt.post_frames,
        post_min_dec=opt.post_min_dec, bounce_px=opt.bounce_px
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

    # 主循环
    try:
        while not exit_signal:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005); continue

            camera_fps = fps_cam.update()

            # 取推理图像（原分辨率）并送入推理线程
            zed.retrieve_image(img_left_net, sl.VIEW.LEFT)
            net_rgba = img_left_net.get_data()
            if net_rgba is not None and net_rgba.size>0:
                if image_queue.full():
                    try: image_queue.get_nowait()
                    except: pass
                image_queue.put(net_rgba.copy())

            # 取检测结果并注入 ZED，检索对象（带 3D）
            dets = None
            try:
                dets = detection_queue.get_nowait()
            except Empty:
                pass
            if dets and len(dets)>0:
                zed.ingest_custom_box_objects(dets)  # 与示例相同的自定义框注入:contentReference[oaicite:5]{index=5}
            zed.retrieve_objects(objs, obj_runtime)

            # 取显示图像（按 display_res）
            zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            frame = img_left.get_data()
            if frame is None: continue
            img = frame.copy()

            # 渲染 ROI/起点线
            render_overlay(img, counter)

            # 处理对象
            event_info = None
            for obj in objs.object_list:
                if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue
                # bbox 到显示坐标
                bbox = obj.bounding_box_2d  # 原始分辨率坐标
                rect_disp = [(int(bbox[i][0]*image_scale[0]), int(bbox[i][1]*image_scale[1])) for i in range(4)]
                # 中心与上边线中心（显示坐标）
                cx = (rect_disp[0][0] + rect_disp[2][0]) * 0.5
                cy = (rect_disp[0][1] + rect_disp[2][1]) * 0.5
                top_y = (rect_disp[0][1] + rect_disp[1][1]) * 0.5
                # 深度
                depth_m = float(obj.position[2]) if np.isfinite(obj.position[2]) else None

                # 状态更新
                ev = counter.update(cx, cy, top_y, depth_m, rect_disp)
                if ev is not None:
                    event_info = ev

                # 画框/中心/深度
                cv2.rectangle(img, rect_disp[0], rect_disp[2], (0,255,0), 2)
                cv2.circle(img, (int(cx),int(cy)), 4, (255,0,0), -1)
                label = f"D={depth_m:.2f}m" if depth_m is not None else "D=--"
                cv2.putText(img, label, (rect_disp[0][0], rect_disp[0][1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # 只处理一个球（默认一个目标）
                break

            # 事件提示
            if event_info:
                if event_info["event"]=="goal":
                    txt,color = "GOAL +1", (0,255,255)
                else:
                    txt,color = "OUT", (0,0,255)
                cv2.putText(img, txt, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 状态文本
            cv2.putText(img, f"Status: {counter.exam_state.value} | State: {counter.shoot_state.value}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.putText(img, f"Score: {counter.goal_count}  Shots: {counter.shot_total}",
                        (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
            cv2.putText(img, f"CamFPS:{camera_fps:.1f}  InfFPS:{inference_fps:.1f}",
                        (20, display_res.height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)

            cv2.imshow("Soccer Shoot Counter", img)

            # 键盘：c 开始；s 中止清零；q/esc 退出
            key = cv2.waitKey(1) & 0xFF
            if key==27 or key==ord('q') or key==ord('Q'):
                break
            elif key==ord('c') or key==ord('C'):
                counter.start()
            elif key==ord('s') or key==ord('S'):
                counter.stop_and_reset()

    except KeyboardInterrupt:
        pass
    finally:
        exit_signal = True
        infer_thread.join(timeout=2.0)
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        main()