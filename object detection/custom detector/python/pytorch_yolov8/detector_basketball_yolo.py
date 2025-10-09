#!/usr/bin/env python3
"""
Basketball Shooting Detection System (YOLO Track Version)
Detects and counts basketball shots using ROI regions with YOLO tracking
No depth information required
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
detection_queue = Queue(maxsize=4)
exit_signal = False
inference_fps = 0.0
camera_fps = 0.0

# Exam related parameters
class ExamState(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    FINISHED = "Finished"

# Basketball shooting state machine
class ShootingState(Enum):
    PREPARE = "Prepare"
    INTERSECTED = "Intersected"  # Ball intersecting with backboard ROI
    FINISHED = "Finished"

# Basketball exam parameters
EXAM_DURATION = 60  # 1 minute exam
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
        x1, y1, x2, y2 = bbox
        bbox_x_min = min(x1, x2)
        bbox_x_max = max(x1, x2)
        bbox_y_min = min(y1, y2)
        bbox_y_max = max(y1, y2)
        
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
    
    def update(self, bbox_display, label=None):
        """Update counter with new detection
        bbox_display: (x1, y1, x2, y2) in display coordinates
        """
        if self.exam_state != ExamState.RUNNING or bbox_display is None:
            return
        
        x1, y1, x2, y2 = bbox_display
        
        # Get ball center
        ball_center_x = (x1 + x2) / 2
        ball_center_y = (y1 + y2) / 2
        ball_center = (ball_center_x, ball_center_y)
        
        # State machine logic
        if self.shooting_state == ShootingState.PREPARE:
            # Check if ball intersects with backboard ROI
            if self.roi_system.check_bbox_intersection(bbox_display, 'backboard'):
                self.shooting_state = ShootingState.INTERSECTED
                self.current_shot_data = []
                print(f"开始记录投篮数据")
        
        elif self.shooting_state == ShootingState.INTERSECTED:
            # Check if ball center is still in backboard ROI
            if self.roi_system.check_point_in_roi(ball_center, 'backboard'):
                # Record data
                self.current_shot_data.append({
                    'center': ball_center,
                    'bbox': bbox_display,
                    'time': time(),
                    'label': label
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
        
        # Check for ball_in label
        ball_in_detected = False
        for i in range(b_intersection_idx + 1, len(self.current_shot_data)):
            if self.current_shot_data[i].get('label') == 'ball_in':
                ball_in_detected = True
                break
        
        if ball_in_detected:
            self.shot_count += 1
            print(f"Ball In! Score: {self.shot_count}")
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
        
        # All conditions met - score!
        self.shot_count += 1
        print(f"投篮进球！当前得分：{self.shot_count}")
    
    def get_status_text(self):
        """Get status text for display"""
        status_lines = []
        
        if self.exam_state == ExamState.IDLE:
            status_lines.append("Status: Waiting")
            status_lines.append("Press 'Q' to start")
        elif self.exam_state == ExamState.RUNNING:
            status_lines.append("Exam in progress")
            status_lines.append(f"Time left: {int(self.exam_remaining_time)}s")
            status_lines.append(f"Score: {self.shot_count}")
            status_lines.append(f"Shooting State: {self.shooting_state.value}")
        elif self.exam_state == ExamState.FINISHED:
            status_lines.append("Exam finished")
            status_lines.append(f"Final Score: {self.shot_count}")
            status_lines.append("Press 'Q' to restart")
        
        return status_lines

# Global instances
basketball_counter = None
roi_system = None

def torch_thread(weights, img_size, conf_thres, iou_thres, class_names, tracker_type='bytetrack'):
    """YOLO inference thread with tracking"""
    global exit_signal, inference_fps
    
    # Check if model file exists
    if not os.path.exists(weights):
        print(f"Error: Model file not found at {weights}")
        exit_signal = True
        return

    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit_signal = True
        return
    
    # Parse positive class names
    positive = set([n.strip().lower() for n in class_names.split(",") if n.strip()])
    
    inference_fps_counter = FPSCounter(window_size=30)
    
    # Track ID storage
    tracked_id = None
    
    while not exit_signal:
        try:
            img_bgr = image_queue.get(timeout=0.1)
        except Empty:
            continue
        
        try:
            # Use track method for detection and tracking
            results = model.track(img_bgr, 
                                persist=True,
                                tracker=f"{tracker_type}.yaml",
                                conf=conf_thres, 
                                iou=iou_thres,
                                imgsz=img_size,
                                verbose=False)
            
            # Process tracking results
            tracked_objects = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    ids = boxes.id
                    
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                        
                        if label.lower() in positive:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # Get tracking ID
                            track_id = None
                            if ids is not None and i < len(ids):
                                track_id = int(ids[i])
                            
                            tracked_objects.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'conf': confidence,
                                'class': cls_id,
                                'label': label,
                                'track_id': track_id
                            })
            
            # Select object to track
            selected_object = None
            
            if len(tracked_objects) > 0:
                # Prefer same ID if tracking
                if tracked_id is not None:
                    for obj in tracked_objects:
                        if obj['track_id'] == tracked_id:
                            selected_object = obj
                            break
                
                # Otherwise select highest confidence
                if selected_object is None:
                    selected_object = max(tracked_objects, key=lambda x: x['conf'])
                    if selected_object['track_id'] is not None:
                        tracked_id = selected_object['track_id']
                    else:
                        tracked_id = None
            else:
                tracked_id = None
            
            inference_fps = inference_fps_counter.update()
            
            # Put result in queue
            if detection_queue.full():
                try:
                    detection_queue.get_nowait()
                except:
                    pass
            detection_queue.put(selected_object)
            
        except Exception as e:
            print(f"Inference Error: {e}")

def render_basketball_view(image, image_scale, tracked_obj):
    """Render basketball detection view"""
    global basketball_counter, roi_system
    
    # Draw ROI regions
    if roi_system and roi_system.is_calibrated:
        roi_system.draw_rois(image)
    
    # Process detected object
    bbox_display = None
    if tracked_obj is not None:
        x1, y1, x2, y2 = tracked_obj['bbox']
        
        # Scale to display resolution
        x1_disp = int(x1 * image_scale[0])
        y1_disp = int(y1 * image_scale[1])
        x2_disp = int(x2 * image_scale[0])
        y2_disp = int(y2 * image_scale[1])
        
        bbox_display = (x1_disp, y1_disp, x2_disp, y2_disp)
        
        # Update counter
        if basketball_counter:
            basketball_counter.update(bbox_display, tracked_obj['label'])
        
        # Draw bounding box
        color = (0, 255, 0)
        if basketball_counter and basketball_counter.shooting_state == ShootingState.INTERSECTED:
            color = (0, 255, 255)
        
        cv2.rectangle(image, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
        
        # Draw center point
        center_x = (x1_disp + x2_disp) // 2
        center_y = (y1_disp + y2_disp) // 2
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Display label and tracking info
        label_text = f"{tracked_obj['label']} ({tracked_obj['conf']:.2f})"
        if tracked_obj['track_id'] is not None:
            label_text += f" ID:{tracked_obj['track_id']}"
        
        cv2.putText(image, label_text,
                   (x1_disp, y1_disp - 10),
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
        'iou_thres': opt.iou_thres,
        'class_names': opt.class_names,
        'tracker_type': opt.tracker
    })
    capture_thread.start()
    
    print("Initializing Camera...")
    zed = sl.Camera()
    
    # Initialize parameters - depth disabled
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth completely
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD720
    
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
    
    print("\n篮球投篮计数系统 (YOLO Track版本)")
    print("操作说明:")
    print("- 按 'Q' 开始/结束考试")
    print("- 按 'ESC' 退出程序")
    print(f"\n投篮规则:")
    print(f"- 考试时间: 60秒")
    print(f"- 系统自动判断进球\n")
    
    # Create window
    cv2.namedWindow("Basketball Shot Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Basketball Shot Counter", display_resolution.width, display_resolution.height)
    
    # Detection storage
    tracked_obj = None
    
    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Update capture FPS
                camera_fps = fps_counter.update()

                # Get image for inference
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                
                # Convert RGBA to BGR for YOLO
                if image_net is not None and image_net.size > 0:
                    image_bgr = cv2.cvtColor(image_net, cv2.COLOR_RGBA2BGR)
                    try:
                        if image_queue.full():
                            image_queue.get_nowait()
                        image_queue.put(image_bgr.copy())
                    except:
                        pass
                
                # Get detection results
                try:
                    tracked_obj = detection_queue.get_nowait()
                except Empty:
                    pass
                
                # Get display image
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                
                # Copy image data to OpenCV format
                image_data_gpu = image_left.get_data()
                if image_data_gpu is not None:
                    np.copyto(image_left_ocv, image_data_gpu)
                    
                    # Render view
                    render_basketball_view(image_left_ocv, image_scale, tracked_obj)
                    
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
    parser.add_argument('--class_names', type=str, default='ball_out,ball_in', 
                        help='comma separated class names to detect')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                        help='Tracker type: bytetrack or botsort')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()
