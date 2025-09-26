#!/usr/bin/env python3
"""
Basketball Shooting Detection System
Detects and counts basketball shots using ROI regions (backboard, above hoop, net)
"""

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
import json
import os

# Global queues for thread communication
image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=2)
exit_signal = False
inference_fps = 0.0

# Exam related parameters
class ExamState(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    FINISHED = "Finished"

# Basketball shooting state machine
class ShootingState(Enum):
    PREPARE = "Prepare"
    INTERSECTED = "Intersected"  # Ball intersecting with backboard ROI, collecting data
    FINISHED = "Finished"

# Basketball exam parameters
EXAM_DURATION = 60  # 1 minute exam
MIN_DEPTH_THRESHOLD = 6.0  # Minimum depth to start tracking (meters)
SCORING_DEPTH_MIN = 6.8  # Minimum depth for valid score (meters)
SCORING_DEPTH_MAX = 7.0  # Maximum depth for valid score (meters)
CONSECUTIVE_FRAMES_REQUIRED = 3  # Consecutive frames required inside ROI

class FPSCounter:
    """FPS calculation utility"""
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

class BasketballROISystem:
    """Basketball ROI management system"""
    def __init__(self):
        self.roi_rects = {
            'backboard': None,   # A - Backboard ROI (largest)
            'above_hoop': None,  # B - Above hoop ROI
            'net': None          # C - Net ROI
        }
        self.roi_polygons = {
            'backboard': None,
            'above_hoop': None,
            'net': None
        }
        self.is_calibrated = False
        
    def load_calibration(self, filename="basketball_roi_calibration.json"):
        """Load ROI calibration data"""
        if not os.path.exists(filename):
            print(f"Error: Calibration file {filename} not found!")
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load ROI rectangles
            self.roi_rects = data.get('roi_rects', {})
            
            # Load ROI polygons
            roi_points = data.get('roi_points', {})
            for roi_name, points in roi_points.items():
                if all(points[corner] is not None for corner in ['top_left', 'top_right', 'bottom_right', 'bottom_left']):
                    self.roi_polygons[roi_name] = np.array([
                        points['top_left'],
                        points['top_right'],
                        points['bottom_right'],
                        points['bottom_left']
                    ], np.int32)
            
            self.is_calibrated = data.get('is_calibrated', False)
            
            # Verify ROI containment
            if not self.verify_roi_containment():
                return False
            
            print(f"Basketball ROI calibration loaded from {filename}")
            print(f"ROI Rects: {self.roi_rects}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def verify_roi_containment(self):
        """Verify that B and C ROIs are contained within A ROI"""
        if not self.is_calibrated:
            return False
        
        # Get backboard (A) bounds
        backboard_rect = self.roi_rects.get('backboard')
        if backboard_rect is None:
            print("Error: Backboard ROI not found!")
            return False
        
        a_x, a_y, a_w, a_h = backboard_rect
        
        # Check above_hoop (B) ROI
        above_hoop_rect = self.roi_rects.get('above_hoop')
        if above_hoop_rect is None:
            print("Error: Above hoop ROI not found!")
            return False
        
        b_x, b_y, b_w, b_h = above_hoop_rect
        if not (b_x >= a_x and b_y >= a_y and 
                b_x + b_w <= a_x + a_w and b_y + b_h <= a_y + a_h):
            print("Error: Above hoop ROI is not contained within backboard ROI!")
            print("Please recalibrate the system.")
            return False
        
        # Check net (C) ROI
        net_rect = self.roi_rects.get('net')
        if net_rect is None:
            print("Error: Net ROI not found!")
            return False
        
        c_x, c_y, c_w, c_h = net_rect
        if not (c_x >= a_x and c_y >= a_y and 
                c_x + c_w <= a_x + a_w and c_y + c_h <= a_y + a_h):
            print("Error: Net ROI is not contained within backboard ROI!")
            print("Please recalibrate the system.")
            return False
        
        print("ROI containment verification passed.")
        return True
    
    def check_bbox_intersection(self, bbox, roi_name):
        """Check if bounding box intersects with ROI"""
        if not self.is_calibrated or roi_name not in self.roi_rects:
            return False
        
        roi_rect = self.roi_rects[roi_name]
        if roi_rect is None:
            return False
        
        roi_x, roi_y, roi_w, roi_h = roi_rect
        
        # Get bbox bounds
        bbox_x_min = min(bbox[0][0], bbox[2][0])
        bbox_x_max = max(bbox[0][0], bbox[2][0])
        bbox_y_min = min(bbox[0][1], bbox[2][1])
        bbox_y_max = max(bbox[0][1], bbox[2][1])
        
        # Check intersection
        return not (bbox_x_max < roi_x or 
                   bbox_x_min > roi_x + roi_w or
                   bbox_y_max < roi_y or
                   bbox_y_min > roi_y + roi_h)
    
    def check_point_in_roi(self, point, roi_name):
        """Check if point is inside ROI polygon"""
        if not self.is_calibrated or roi_name not in self.roi_polygons:
            return False
        
        polygon = self.roi_polygons[roi_name]
        if polygon is None:
            return False
        
        return cv2.pointPolygonTest(polygon, tuple(point), False) >= 0
    
    def draw_rois(self, image):
        """Draw all ROI regions on image"""
        if not self.is_calibrated:
            return
        
        colors = {
            'backboard': (0, 255, 255),    # Yellow
            'above_hoop': (0, 255, 0),      # Green
            'net': (0, 0, 255)              # Red
        }
        
        for roi_name, polygon in self.roi_polygons.items():
            if polygon is not None:
                cv2.polylines(image, [polygon], True, colors[roi_name], 2)
                
                # Add label
                rect = self.roi_rects[roi_name]
                if rect:
                    cv2.putText(image, roi_name.upper(), 
                               (rect[0], rect[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[roi_name], 1)

class BasketballCounter:
    """Basketball shot counter with ROI-based detection"""
    def __init__(self, roi_system):
        self.roi_system = roi_system
        
        # Shooting detection state
        self.shooting_state = ShootingState.PREPARE
        self.shot_count = 0
        
        # Data collection for current shot
        self.current_shot_data = []
        
        # Exam state
        self.exam_state = ExamState.IDLE
        self.exam_start_time = 0
        self.exam_remaining_time = EXAM_DURATION
        
    def start_exam(self):
        """Start the exam"""
        self.exam_state = ExamState.RUNNING
        self.exam_start_time = time()
        self.exam_remaining_time = EXAM_DURATION
        self.shot_count = 0
        self.shooting_state = ShootingState.PREPARE
        self.current_shot_data = []
        print("\n考试开始！时间：60秒")
    
    def finish_exam(self):
        """Finish the exam"""
        self.exam_state = ExamState.FINISHED
        print(f"\n考试结束！投进球数：{self.shot_count}")
    
    def update_exam_time(self):
        """Update exam remaining time"""
        if self.exam_state == ExamState.RUNNING:
            elapsed = time() - self.exam_start_time
            self.exam_remaining_time = max(0, EXAM_DURATION - elapsed)
            
            if self.exam_remaining_time <= 0:
                self.finish_exam()
    
    def update(self, bbox, position):
        """Update counter with new detection"""
        if self.exam_state != ExamState.RUNNING:
            return
        
        # Get ball center and depth
        ball_center_x = (bbox[0][0] + bbox[2][0]) / 2
        ball_center_y = (bbox[0][1] + bbox[2][1]) / 2
        ball_center = (ball_center_x, ball_center_y)
        ball_depth = position[2] if np.isfinite(position[2]) else 0.0
        
        # State machine logic
        if self.shooting_state == ShootingState.PREPARE:
            # Check if ball intersects with backboard ROI and depth > 6m
            if (self.roi_system.check_bbox_intersection(bbox, 'backboard') and 
                ball_depth > MIN_DEPTH_THRESHOLD):
                self.shooting_state = ShootingState.INTERSECTED
                self.current_shot_data = []
                print(f"开始记录投篮数据 (深度: {ball_depth:.2f}m)")
        
        elif self.shooting_state == ShootingState.INTERSECTED:
            # Check if ball center is still in backboard ROI
            if self.roi_system.check_point_in_roi(ball_center, 'backboard'):
                # Record data
                self.current_shot_data.append({
                    'center': ball_center,
                    'depth': ball_depth,
                    'bbox': bbox,
                    'time': time()
                })
            else:
                # Ball left backboard ROI, analyze the shot
                self.shooting_state = ShootingState.FINISHED
                self.analyze_shot()
                self.shooting_state = ShootingState.PREPARE
    
    def analyze_shot(self):
        """Analyze collected data to determine if shot was made"""
        if len(self.current_shot_data) < 10:  # Not enough data
            print("投篮数据不足，判定为出界")
            return
        
        # Check conditions for scoring
        b_intersection_idx = -1  # Index when ball intersects with B
        c_intersection_idx = -1  # Index when ball intersects with C
        
        # Find B intersection
        for i, data in enumerate(self.current_shot_data):
            if self.roi_system.check_bbox_intersection(data['bbox'], 'above_hoop'):
                b_intersection_idx = i
                break
        
        # Check for consecutive frames in B after intersection
        b_condition_met = False
        if b_intersection_idx >= 0:
            consecutive_in_b = 0
            for i in range(b_intersection_idx + 1, len(self.current_shot_data)):
                if self.roi_system.check_point_in_roi(self.current_shot_data[i]['center'], 'above_hoop'):
                    consecutive_in_b += 1
                    if consecutive_in_b >= CONSECUTIVE_FRAMES_REQUIRED:
                        b_condition_met = True
                        break
                else:
                    consecutive_in_b = 0
        
        if not b_condition_met:
            print("未满足条件B (篮筐上方)，判定为出界")
            return
        
        # Find C intersection (must be after B)
        for i in range(b_intersection_idx + 1, len(self.current_shot_data)):
            if self.roi_system.check_bbox_intersection(self.current_shot_data[i]['bbox'], 'net'):
                c_intersection_idx = i
                break
        
        # Check for consecutive frames in C after intersection
        c_condition_met = False
        if c_intersection_idx >= 0:
            consecutive_in_c = 0
            for i in range(c_intersection_idx + 1, len(self.current_shot_data)):
                if self.roi_system.check_point_in_roi(self.current_shot_data[i]['center'], 'net'):
                    consecutive_in_c += 1
                    if consecutive_in_c >= CONSECUTIVE_FRAMES_REQUIRED:
                        c_condition_met = True
                        break
                else:
                    consecutive_in_c = 0
        
        if not c_condition_met:
            print("未满足条件C (篮网)，判定为出界")
            return
        
        # Check depth requirement
        depth_valid = True
        for data in self.current_shot_data[b_intersection_idx:]:
            if not (SCORING_DEPTH_MIN <= data['depth'] <= SCORING_DEPTH_MAX):
                depth_valid = False
                break
        
        if not depth_valid:
            print(f"深度不在有效范围 ({SCORING_DEPTH_MIN}-{SCORING_DEPTH_MAX}m)，判定为出界")
            return
        
        # All conditions met - score!
        self.shot_count += 1
        print(f"投篮进球！当前得分：{self.shot_count}")
    
    def get_status_text(self):
        """Get status text for display"""
        status_lines = []
        
        if self.exam_state == ExamState.IDLE:
            status_lines.append("状态: 等待开始")
            status_lines.append("按 'Q' 开始考试")
        elif self.exam_state == ExamState.RUNNING:
            status_lines.append(f"考试进行中")
            status_lines.append(f"剩余时间: {int(self.exam_remaining_time)}秒")
            status_lines.append(f"投进球数: {self.shot_count}")
            status_lines.append(f"投篮状态: {self.shooting_state.value}")
        elif self.exam_state == ExamState.FINISHED:
            status_lines.append("考试结束")
            status_lines.append(f"最终得分: {self.shot_count}")
            status_lines.append("按 'Q' 重新开始")
        
        return status_lines

# Global instances
basketball_counter = None
roi_system = None

def xywh2abcd(xywh, im_shape):
    """Convert YOLO format to custom box format"""
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
    """Convert detections to custom box format"""
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        
        if label == "basketball":
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
            obj.label = cls_id
            obj.probability = float(det.conf[0])
            obj.is_grounded = False
            obj.unique_object_id = sl.generate_unique_id()
            output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.25, iou_thres=0.55):
    """YOLO inference thread"""
    global exit_signal, inference_fps
    
    # Check if model file exists
    if not os.path.exists(weights):
        print(f"Error: Model file not found at {weights}")
        print(f"Please ensure the model file is placed at the correct location")
        exit_signal = True
        return

    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit_signal = True
        return
    
    inference_fps_counter = FPSCounter(window_size=30)
    
    while not exit_signal:
        try:
            image_data = image_queue.get(timeout=0.1)
            img_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
            results = model.predict(img_rgb, save=False, imgsz=img_size, verbose=False, 
                                   conf=conf_thres, iou=iou_thres)[0]
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
            print(f"Inference Error: {e}")

def render_basketball_view(image, image_scale, objects):
    """Render basketball detection view"""
    global basketball_counter, roi_system
    
    # Draw ROI regions
    if roi_system and roi_system.is_calibrated:
        roi_system.draw_rois(image)
    
    # Process detected objects
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # Get bounding box
            bbox = obj.bounding_box_2d
            
            # Update counter
            if basketball_counter:
                basketball_counter.update(bbox, obj.position)
            
            # Draw bounding box
            color = (0, 255, 0)
            thickness = 2
            
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         color, thickness)
            
            # Draw center point
            center_x = int((bbox[0][0] + bbox[2][0]) / 2 * image_scale[0])
            center_y = int((bbox[0][1] + bbox[2][1]) / 2 * image_scale[1])
            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Display depth information
            distance = obj.position[2] if np.isfinite(obj.position[2]) else 0.0
            label_text = f"D={distance:.2f}m"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display status information
    if basketball_counter:
        basketball_counter.update_exam_time()
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Display status text
        status_lines = basketball_counter.get_status_text()
        y_offset = 30
        for line in status_lines:
            color = (255, 255, 255)
            cv2.putText(image, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Display FPS information
        cv2.putText(image, f"Camera FPS: {camera_fps:.1f} | Inference FPS: {inference_fps:.1f}",
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Global FPS variable
camera_fps = 0.0

def main():
    global exit_signal, basketball_counter, camera_fps, inference_fps, roi_system
    
    # Check model file before starting
    if not os.path.exists(opt.weights):
        print(f"Error: Model file not found at {opt.weights}")
        print("Please specify correct model path using --weights argument")
        return
    
    # Create and load ROI system
    roi_system = BasketballROISystem()
    
    # Load calibration data - exit if failed
    if not roi_system.load_calibration():
        print("\nError: Basketball ROI calibration data not found or invalid!")
        print("Please run reference_system_basketball.py first to calibrate ROI regions")
        return
    
    # Create counter
    basketball_counter = BasketballCounter(roi_system)
    
    # Start inference thread
    capture_thread = Thread(target=torch_thread, kwargs={
        'weights': opt.weights, 
        'img_size': opt.img_size, 
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres
    })
    capture_thread.start()
    
    print("Initializing Camera...")
    zed = sl.Camera()
    
    # Initialize parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10
    init_params.depth_minimum_distance = 2.5
    init_params.depth_stabilization = 10
    
    # Handle input source
    if opt.ip is not None:
        print(f"Connecting to remote stream at {opt.ip}")
        try:
            ip_address, port = opt.ip.split(':')
            port = int(port)
            init_params.set_from_stream(ip_address, port)
        except ValueError:
            print(f"Invalid IP format. Please use format: IP:PORT")
            exit()
    elif opt.svo is not None:
        input_type = sl.InputType()
        input_type.set_from_svo_file(opt.svo)
        init_params.input = input_type
        init_params.svo_real_time_mode = False
    
    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()
    
    image_left_tmp = sl.Mat()
    print("Initialized Camera")
    
    # Enable position tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Configure object detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    obj_param.max_range = 10
    
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Enable object detection failed: {repr(err)}. Exit program.")
        zed.close()
        exit()
    
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # Display settings
    camera_info = zed.get_camera_information()
    camera_res = camera_info.camera_configuration.resolution
    
    # Display resolution
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    
    # Create display image
    image_left_ocv = np.zeros((display_resolution.height, display_resolution.width, 4), np.uint8)
    image_left = sl.Mat()
    
    # FPS counter
    fps_counter = FPSCounter(window_size=30)
    
    print("\n篮球投篮计数系统")
    print("操作说明:")
    print("- 按 'Q' 开始/结束考试")
    print("- 按 'ESC' 退出程序")
    print(f"\n投篮规则:")
    print(f"- 在2.5米线外投篮")
    print(f"- 考试时间: 60秒")
    print(f"- 系统自动判断进球\n")
    
    # Create window
    cv2.namedWindow("Basketball Shot Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Basketball Shot Counter", display_resolution.width, display_resolution.height)
    
    # Detection storage
    detections = []
    
    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Update capture FPS
                camera_fps = fps_counter.update()

                # Get image for inference
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                
                # Copy image data to queue
                if image_net is not None and image_net.size > 0:
                    try:
                        if image_queue.full():
                            image_queue.get_nowait()
                        image_queue.put(image_net.copy())
                    except:
                        pass
                
                # Get detection results
                try:
                    detections = detection_queue.get_nowait()
                except Empty:
                    pass
                
                # Inject detections to ZED
                if detections and len(detections) > 0:
                    zed.ingest_custom_box_objects(detections)
                
                # Get tracked objects
                zed.retrieve_objects(objects, obj_runtime_param)
                
                # Get display image
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                
                # Copy image data to OpenCV format
                image_data_gpu = image_left.get_data()
                if image_data_gpu is not None:
                    np.copyto(image_left_ocv, image_data_gpu)
                    
                    # Render view
                    render_basketball_view(image_left_ocv, image_scale, objects)
                    
                    # Display image
                    cv2.imshow("Basketball Shot Counter", image_left_ocv)
                
                # Keyboard control
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    exit_signal = True
                elif key == ord('q') or key == ord('Q'):
                    if basketball_counter.exam_state == ExamState.IDLE or basketball_counter.exam_state == ExamState.FINISHED:
                        basketball_counter.start_exam()
                    elif basketball_counter.exam_state == ExamState.RUNNING:
                        basketball_counter.finish_exam()
            else:
                print("Camera grab failed")
                sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        exit_signal = True
        
    print("Cleaning up resources...")
    capture_thread.join(timeout=2.0)
    zed.close()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/basketball.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()