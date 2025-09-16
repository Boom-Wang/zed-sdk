#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from collections import deque
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

# ----------------------------
# 线程队列（与项目一致风格）
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
    """将YOLO检测结果转换为ZED自定义框（仅保留足球类）"""
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
    """YOLO 推理线程：模型->预测->CustomBox->队列（工程同款流程）"""
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

# 计算两点式直线在给定 y 处的 x（延长线也成立）
def x_on_line_at_y(p1, p2, y):
    if y is None:
        return None
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
    # rect: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    for r in rect:
        if point_in_poly((r[0],r[1]), poly):
            return True
    x_min = min([r[0] for r in rect]); x_max = max([r[0] for r in rect])
    y_min = min([r[1] for r in rect]); y_max = max([r[1] for r in rect])
    for p in poly:
        if x_min<=p[0]<=x_max and y_min<=p[1]<=y_max:
            return True
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
# 业务逻辑：射门计数器（含反弹&遮挡优化）
# ----------------------------
class SoccerCounter:
    def __init__(self, goal_poly_disp, start_p1_disp, start_p2_disp,
                 depth_min=3.0, depth_max=4.2,
                 pre_frames=10, post_frames=5, post_min_dec=3,
                 bounce_px=4.0,
                 disappear_goal_frames=3,        # 遮挡后判进球的连续消失帧数
                 post_window_for_disappear=5,    # 相交后用于趋势判断的窗口
                 miss_timeout_frames=40          # Shooting 阶段长时间丢失的超时（出界）
                 ):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0

        self.goal_poly = goal_poly_disp
        self.start_p1 = start_p1_disp
        self.start_p2 = start_p2_disp
        # A(左上),B(左下),C(右下),D(右上)
        self.crossbar_Dy = goal_poly_disp[3][1]

        # 参数
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.post_min_dec = post_min_dec
        self.bounce_px = bounce_px
        self.disappear_goal_frames = disappear_goal_frames
        self.post_window_for_disappear = post_window_for_disappear
        self.miss_timeout_frames = miss_timeout_frames

        # 历史缓存：(t,cx,cy,top_y,depth,rect,visible)
        self.hist = deque(maxlen=300)
        self.active_inter_idx = None   # ROI 相交帧索引（锚点）
        self.await_decision = False

        # 遮挡统计
        self.missing_since_inter = 0
        self.shooting_missing = 0

        # 鲁棒入态
        self.left_side_streak = 0

    # ---------- 基础工具 ----------
    def _get_last_visible_before_current(self):
        """返回最近一帧可见帧 (cx,cy)；若不存在，返回 None"""
        if len(self.hist) < 2:
            return None
        for k in range(len(self.hist)-2, -1, -1):
            if self.hist[k][6]:
                return (self.hist[k][1], self.hist[k][2])
        return None

    def _is_right_of_startline(self, cx, cy):
        x_line = x_on_line_at_y(self.start_p1, self.start_p2, cy)
        if x_line is None:
            return False
        return cx > x_line + 2.0

    def _is_left_of_startline(self, cx, cy):
        x_line = x_on_line_at_y(self.start_p1, self.start_p2, cy)
        if x_line is None:
            return False
        return cx <= x_line + 2.0

    def _crossed_startline_to_left(self, prev, curr):
        if prev is None or curr is None:
            return False
        cx0, cy0 = prev; cx1, cy1 = curr
        if cy0 is None or cy1 is None:
            return False
        x_line0 = x_on_line_at_y(self.start_p1, self.start_p2, cy0)
        x_line1 = x_on_line_at_y(self.start_p1, self.start_p2, cy1)
        if x_line0 is None or x_line1 is None:
            return False
        return (cx0 > x_line0 + 2.0) and (cx1 <= x_line1 + 2.0) and (cx1 < cx0 - 1.0)

    def _cx_trend_count_left(self, xs):
        cnt = 0
        for i in range(1, len(xs)):
            if xs[i] < xs[i-1] - 1.0:
                cnt += 1
        return cnt

    def _depth_ok_majority(self, depths):
        vals = [d for d in depths if d is not None and np.isfinite(d)]
        if not vals: return False
        ok = sum(1 for d in vals if self.depth_min <= d <= self.depth_max)
        return ok >= max(1, len(vals)//2)

    # ---------- 状态控制 ----------
    def start(self):
        self.exam_state = ExamState.RUNNING
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self.hist.clear()
        self.active_inter_idx = None
        self.await_decision = False
        self.missing_since_inter = 0
        self.shooting_missing = 0
        self.left_side_streak = 0

    def stop_and_reset(self):
        self.exam_state = ExamState.IDLE
        self.shoot_state = ShootState.PREPARE
        self.goal_count = 0
        self.shot_total = 0
        self.hist.clear()
        self.active_inter_idx = None
        self.await_decision = False
        self.missing_since_inter = 0
        self.shooting_missing = 0
        self.left_side_streak = 0

    # ---------- 最终判定（可见后帧足够时触发） ----------
    def _decide_goal_or_out(self):
        if self.active_inter_idx is None:
            return False

        L = len(self.hist); i0 = self.active_inter_idx

        # **改动1：用“前后帧窗口”综合判断**
        pre_vis  = [e for e in list(self.hist)[max(0, i0-self.pre_frames):i0] if e[6]]
        post_vis = [e for e in list(self.hist)[i0:min(L, i0+self.post_frames+1)] if e[6]]

        # 趋势：主要看相交后的继续向左，容许轻微抖动
        post_xs = [e[1] for e in post_vis]
        left_trend_ok = self._cx_trend_count_left(post_xs) >= self.post_min_dec

        # 快速反弹：相交后出现连续两帧明显右移
        rebound = False
        if len(post_xs) >= 3:
            for k in range(2, len(post_xs)):
                dx1 = post_xs[k-1] - post_xs[k-2]
                dx2 = post_xs[k]   - post_xs[k-1]
                if dx1 > self.bounce_px and dx2 > self.bounce_px:
                    rebound = True
                    break

        # 深度：优先用 post，多数派；若 post 没深度，回退到 pre
        depths_post = [e[4] for e in post_vis if e[4] is not None]
        depths_pre  = [e[4] for e in pre_vis  if e[4] is not None]
        depth_ok = self._depth_ok_majority(depths_post) or self._depth_ok_majority(depths_pre)

        # 横梁：post 中所有可见帧 top_y 不小于 D_y；若 post 无可见 top_y，则回退到相交帧或 pre 最后可见
        topys_post = [e[3] for e in post_vis if e[3] is not None]
        if len(topys_post) > 0:
            crossbar_ok = min(topys_post) >= self.crossbar_Dy
        else:
            # 回退：相交帧的 top_y 或 pre 最后可见的 top_y
            ty_candidates = []
            if self.hist[i0][6] and self.hist[i0][3] is not None:
                ty_candidates.append(self.hist[i0][3])
            if len(pre_vis) > 0 and pre_vis[-1][3] is not None:
                ty_candidates.append(pre_vis[-1][3])
            crossbar_ok = (len(ty_candidates) > 0) and (min(ty_candidates) >= self.crossbar_Dy)

        result = False
        if (not rebound) and left_trend_ok and depth_ok and crossbar_ok:
            self.goal_count += 1
            result = True

        # 结束一脚
        self.shoot_state = ShootState.FINISHED
        self.await_decision = False
        self.active_inter_idx = None
        self.missing_since_inter = 0
        self.shooting_missing = 0
        return result

    # ---------- 遮挡辅助：相交后很快消失时触发 ----------
    def _decide_goal_by_disappear(self):
        if self.active_inter_idx is None:
            return False

        L = len(self.hist); i0 = self.active_inter_idx
        pre_vis  = [e for e in list(self.hist)[max(0, i0-self.pre_frames):i0] if e[6]]
        post_vis = [e for e in list(self.hist)[i0:min(L, i0+self.post_window_for_disappear+1)] if e[6]]

        # 运动方向：优先用 post 的最后两帧，否则用 pre 最后一帧到相交帧
        moving_left = True
        if len(post_vis) >= 2:
            moving_left = (post_vis[-1][1] < post_vis[-2][1] - 1.0)
        elif len(pre_vis) >= 1 and self.hist[i0][6]:
            moving_left = (self.hist[i0][1] < pre_vis[-1][1] - 1.0)

        # 深度：post 或 pre 任一窗口内存在合规深度即可
        depths_post = [e[4] for e in post_vis if e[4] is not None]
        depths_pre  = [e[4] for e in pre_vis  if e[4] is not None]
        depth_ok = self._depth_ok_majority(depths_post) or self._depth_ok_majority(depths_pre)

        # 横梁：post 的 min(top_y) 或者回退到相交帧/ pre 最后可见
        topys_post = [e[3] for e in post_vis if e[3] is not None]
        if len(topys_post) > 0:
            crossbar_ok = min(topys_post) >= self.crossbar_Dy
        else:
            ty_candidates = []
            if self.hist[i0][6] and self.hist[i0][3] is not None:
                ty_candidates.append(self.hist[i0][3])
            if len(pre_vis) > 0 and pre_vis[-1][3] is not None:
                ty_candidates.append(pre_vis[-1][3])
            crossbar_ok = (len(ty_candidates) > 0) and (min(ty_candidates) >= self.crossbar_Dy)

        result = False
        if moving_left and depth_ok and crossbar_ok:
            self.goal_count += 1
            result = True

        # 结束一脚
        self.shoot_state = ShootState.FINISHED
        self.await_decision = False
        self.active_inter_idx = None
        self.missing_since_inter = 0
        self.shooting_missing = 0
        return result

    # ---------- 每帧更新（有检测） ----------
    def update(self, cx_disp, cy_disp, top_y_disp, depth_m, rect_disp):
        if self.exam_state != ExamState.RUNNING:
            return None

        # 记录可见帧
        self.hist.append((time(), float(cx_disp), float(cy_disp),
                          float(top_y_disp) if top_y_disp is not None else None,
                          float(depth_m) if depth_m is not None else None,
                          rect_disp, True))
        self.shooting_missing = 0  # 有检测 -> 清零

        # 找最近一帧“可见”的上一帧
        prev_vis = self._get_last_visible_before_current()
        cx_curr, cy_curr = self.hist[-1][1], self.hist[-1][2]

        if self.shoot_state == ShootState.PREPARE:
            # 右侧即准备；连续在左侧也计数，用于鲁棒入态
            if self._is_left_of_startline(cx_curr, cy_curr):
                self.left_side_streak += 1
            else:
                self.left_side_streak = 0

            # 1) 标准：越线且向左
            crossed = self._crossed_startline_to_left(prev_vis, (cx_curr, cy_curr))
            # 2) 鲁棒：已在左侧且 x 明显减少（防丢失越线瞬间）
            robust = False
            if prev_vis is not None:
                robust = (self.left_side_streak >= 2) and (cx_curr < prev_vis[0] - 1.0)

            if crossed or robust:
                self.shoot_state = ShootState.SHOOTING
                self.shot_total += 1
                self.await_decision = False
                self.missing_since_inter = 0

        elif self.shoot_state == ShootState.SHOOTING:
            # **改动2：相交即触发候选（不在相交帧就卡深度/横梁）**
            if rect_poly_intersect(rect_disp, self.goal_poly) and (not self.await_decision):
                self.active_inter_idx = len(self.hist)-1
                self.await_decision = True
                self.missing_since_inter = 0

            # 若已进入候选：达到最少“可见后帧”数量 或 快速反弹 -> 立即判定
            if self.await_decision:
                i0 = self.active_inter_idx
                vis_xs = [e[1] for e in list(self.hist)[i0:] if e[6]]
                need_decide = (len(vis_xs) >= self.post_frames)

                # 提前检查快速反弹
                if not need_decide and len(vis_xs) >= 3:
                    dx1 = vis_xs[-2]-vis_xs[-3]; dx2 = vis_xs[-1]-vis_xs[-2]
                    if dx1 > self.bounce_px and dx2 > self.bounce_px:
                        need_decide = True

                if need_decide:
                    goal = self._decide_goal_or_out()
                    return {"event":"goal" if goal else "out"}

            # 回到起点线右侧 -> 直接结束为出界
            if self._is_right_of_startline(cx_curr, cy_curr):
                self.shoot_state = ShootState.FINISHED
                self.await_decision = False
                self.active_inter_idx = None
                self.missing_since_inter = 0
                self.shooting_missing = 0
                return {"event":"out"}

        elif self.shoot_state == ShootState.FINISHED:
            if self._is_right_of_startline(cx_curr, cy_curr):
                self.shoot_state = ShootState.PREPARE

        return None

    # ---------- 每帧更新（无检测/丢失） ----------
    def update_missing(self):
        """在当前帧**检测不到球**时调用，用于遮挡辅助判定与超时兜底"""
        if self.exam_state != ExamState.RUNNING:
            return None

        # 记录“不可见帧”
        self.hist.append((time(), None, None, None, None, None, False))

        if self.shoot_state == ShootState.SHOOTING:
            self.shooting_missing += 1

            # 已进入候选：遮挡辅助（相交后连续缺失 N 帧 -> 判进球）
            if self.await_decision:
                self.missing_since_inter += 1
                if self.missing_since_inter >= self.disappear_goal_frames:
                    goal = self._decide_goal_by_disappear()
                    return {"event":"goal" if goal else "out"}

            # 未形成候选且长时间缺失 -> 认为本次结束（出界）
            elif self.shooting_missing >= self.miss_timeout_frames:
                self.shoot_state = ShootState.FINISHED
                self.await_decision = False
                self.active_inter_idx = None
                self.missing_since_inter = 0
                self.shooting_missing = 0
                return {"event":"out"}

        return None

# ----------------------------
# 渲染与主循环
# ----------------------------
def render_overlay(img, counter: SoccerCounter):
    cv2.polylines(img, [np.array(counter.goal_poly, dtype=np.int32)], True, (0,0,255), 2)
    cv2.line(img, counter.start_p1, counter.start_p2, (255,255,0), 2)
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
    # 遮挡/超时参数
    parser.add_argument('--disappear_goal_frames', type=int, default=3)
    parser.add_argument('--post_window_for_disappear', type=int, default=5)
    parser.add_argument('--miss_timeout_frames', type=int, default=40)
    # ROI 基准分辨率（你的标注在 1280x720）
    parser.add_argument('--roi_base_w', type=int, default=1280)
    parser.add_argument('--roi_base_h', type=int, default=720)
    opt = parser.parse_args()

    # 推理线程
    infer_thread = Thread(target=torch_thread, kwargs=dict(
        weights=opt.weights, img_size=opt.img_size,
        conf_thres=opt.conf_thres, iou_thres=opt.iou_thres,
        class_names=opt.class_names
    ))
    infer_thread.start()

    # ---- ZED 相机初始化（与官方/工程示例一致）----
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.camera_fps = 60     # 可改为 40 以匹配你的相机
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # 可改为 HD720
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10.0
    init_params.depth_minimum_distance = 0.5
    init_params.depth_stabilization = 50

    runtime_params = sl.RuntimeParameters()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed"); return

    pose_param = sl.PositionalTrackingParameters()
    pose_param.set_as_static = True
    zed.enable_positional_tracking(pose_param)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)

    # 分辨率与缩放
    cam_info = zed.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    display_res = sl.Resolution(min(cam_res.width, 1280), min(cam_res.height, 720))
    sx = display_res.width / float(opt.roi_base_w)
    sy = display_res.height / float(opt.roi_base_h)
    image_scale = (display_res.width / cam_res.width, display_res.height / cam_res.height)

    # 门框 ROI + 起点线（基于 1280x720 的像素标注）
    A = (92, 528); B=(92,670); C=(250,590); D=(250,494)  # A左上,B左下,C右下,D右上
    S1=(1023,587); S2=(1170,662)
    goal_poly_disp = scale_points([A,B,C,D], sx, sy)
    start_p1_disp, start_p2_disp = scale_points([S1,S2], sx, sy)

    counter = SoccerCounter(
        goal_poly_disp, start_p1_disp, start_p2_disp,
        depth_min=opt.depth_min, depth_max=opt.depth_max,
        pre_frames=opt.pre_frames, post_frames=opt.post_frames,
        post_min_dec=opt.post_min_dec, bounce_px=opt.bounce_px,
        disappear_goal_frames=opt.disappear_goal_frames,
        post_window_for_disappear=opt.post_window_for_disappear,
        miss_timeout_frames=opt.miss_timeout_frames
    )

    # ZED 容器
    img_left = sl.Mat()
    img_left_net = sl.Mat()
    objs = sl.Objects()
    obj_runtime = sl.ObjectDetectionRuntimeParameters()

    # FPS
    fps_cam = FPSCounter(30)

    # UI
    cv2.namedWindow("Soccer Shoot Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Soccer Shoot Counter", display_res.width, display_res.height)

    try:
        while not exit_signal:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005); continue

            camera_fps = fps_cam.update()

            # 1) 送推理图像
            zed.retrieve_image(img_left_net, sl.VIEW.LEFT)
            net_rgba = img_left_net.get_data()
            if net_rgba is not None and net_rgba.size>0:
                if image_queue.full():
                    try: image_queue.get_nowait()
                    except: pass
                image_queue.put(net_rgba.copy())

            # 2) 取推理结果 -> 注入 ZED
            dets = None
            try:
                dets = detection_queue.get_nowait()
            except Empty:
                pass
            if dets and len(dets)>0:
                zed.ingest_custom_box_objects(dets)
            zed.retrieve_objects(objs, obj_runtime)

            # 3) 取显示图像
            zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            frame = img_left.get_data()
            if frame is None: continue
            img = frame.copy()

            # 4) 渲染 ROI/起点线
            render_overlay(img, counter)

            # 5) 处理对象（只取一个球）
            event_info = None
            found = False
            for obj in objs.object_list:
                if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue
                found = True

                # bbox -> 显示坐标
                bbox = obj.bounding_box_2d
                rect_disp = [(int(bbox[i][0]*image_scale[0]), int(bbox[i][1]*image_scale[1])) for i in range(4)]
                # 中心与上边线中心（显示坐标）
                cx = (rect_disp[0][0] + rect_disp[2][0]) * 0.5
                cy = (rect_disp[0][1] + rect_disp[2][1]) * 0.5
                top_y = (rect_disp[0][1] + rect_disp[1][1]) * 0.5
                # 深度（Z 轴）
                depth_m = float(obj.position[2]) if np.isfinite(obj.position[2]) else None

                # 状态更新
                ev = counter.update(cx, cy, top_y, depth_m, rect_disp)
                if ev is not None: event_info = ev

                # 画框/中心/深度
                cv2.rectangle(img, rect_disp[0], rect_disp[2], (0,255,0), 2)
                cv2.circle(img, (int(cx),int(cy)), 4, (255,0,0), -1)
                label = f"D={depth_m:.2f}m" if depth_m is not None else "D=--"
                cv2.putText(img, label, (rect_disp[0][0], rect_disp[0][1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break

            # 若本帧无检测，驱动“遮挡/超时”逻辑
            if not found:
                ev = counter.update_missing()
                if ev is not None: event_info = ev

            # 6) 事件提示
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