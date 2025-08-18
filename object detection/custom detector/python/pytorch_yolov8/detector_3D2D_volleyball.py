#!/usr/bin/env python3
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
INITIAL_HEIGHT = 1.2  # Initial state height threshold (meters) - only used to determine if pass started

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

# Ball state machine
class BallState(Enum):
    INITIAL = "Initial"
    RISING = "Rising"
    FALLING = "Falling"

# Simplified perspective reference system (read-only)
class PrecisePerspectiveSystem:
    def __init__(self):
        self.camera_height = 1.5
        self.bar_height = 2.35
        self.bar_distances = {}
        self.bar_2d_points = {}
        self.distance_to_y_pixel = {}
        self.is_calibrated = False
        
    def load_calibration(self, filename="perspective_calibration.json"):
        """Load calibration data"""
        if not os.path.exists(filename):
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
            
            print(f"Calibration data loaded from {filename}")
            print(f"Bar 2D Points: {self.bar_2d_points}")
            print(f"Distance to Y Pixel Mapping: {self.distance_to_y_pixel}")
            print(f"Is Calibrated: {self.is_calibrated}")
            return True
        except Exception as e:
            print(f"Failed to load calibration data: {e}")
            return False
    
    def check_ball_height_by_perspective(self, ball_depth, ball_bottom_y):
        """
        Use pure perspective method to determine if ball exceeds bar height
        
        Args:
            ball_depth: Ball depth (meters)
            ball_bottom_y: Ball bottom Y pixel coordinate
        
        Returns:
            (is_above, confidence, reference_y)
        """
        if not self.is_calibrated or not self.distance_to_y_pixel:
            return None, 0, None
        
        # Round to nearest 0.1m
        rounded_depth = round(ball_depth, 1)
        
        # Find reference Y coordinate
        reference_y = self.distance_to_y_pixel.get(rounded_depth)
        
        if reference_y is None:
            # If no exact match, find closest
            closest_depth = min(self.distance_to_y_pixel.keys(), 
                               key=lambda x: abs(x - rounded_depth))
            reference_y = self.distance_to_y_pixel[closest_depth]
        
        # Judge: when ball bottom Y coordinate <= reference Y coordinate, it exceeds height
        is_above = ball_bottom_y <= reference_y
        confidence = self._calculate_confidence(ball_depth, ball_bottom_y, reference_y)
        
        return is_above, confidence, reference_y
    
    def _calculate_confidence(self, depth, ball_y, ref_y):
        """Calculate judgment confidence"""
        confidence = 0.5
        
        # Depth reasonableness
        if 3.45 <= depth <= 7.0:
            confidence += 0.3
        elif depth < 3.0 or depth >= 8.0:
            confidence += 0.2
        
        # Y coordinate difference significance
        y_diff = abs(ball_y - ref_y)
        if y_diff > 10:  # Significant pixel difference
            confidence += 0.2
        elif y_diff > 5:
            confidence += 0.1
        return min(confidence, 1.0)
    
    def visualize_reference(self, image, ball_depth=None, ball_y=None, ref_y=None):
        """Draw reference lines on image - only draw connections between 6 calibration points"""
        if not self.is_calibrated:
            return image
        
        # 绘制三根颜色不同的横杆
        colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)] 
        bar_names = ['A', 'B', 'C']
        
        # 绘制其他水平杆
        for i, bar in enumerate(bar_names):
            left = self.bar_2d_points.get(f'{bar}_left')
            right = self.bar_2d_points.get(f'{bar}_right')
            if left and right:
                cv2.line(image, left, right, colors[i], 2)
                # Add bar label
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
        
        # 绘制垂直连接 (右侧)
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
        return image

class PerspectiveVolleyballCounter:
    def __init__(self, reference_system=None):
        self.reference_system = reference_system
        
        # Counting related
        self.count = 0
        self.current_state = BallState.INITIAL
        
        # Depth tracking (for judging rise/fall)
        self.depth_history = deque(maxlen=10)
        self.y_history = deque(maxlen=10)
        
        # Exam state
        self.exam_state = ExamState.IDLE
        self.exam_start_time = 0
        self.exam_remaining_time = EXAM_DURATION
        
        # Mark if just reached peak
        self.just_reached_peak = False
        
        # Current state
        self.current_depth = 0
        self.current_is_above = False
        self.current_confidence = 0
        
    def update(self, ball_position, ball_bottom_y):
        """Update volleyball state and count"""
        if self.exam_state != ExamState.RUNNING:
            return
        
        current_time = time()
        self.current_depth = round(ball_position[2], 1)  # Z axis is depth

        # 使用参考系统判断球的高度
        if self.reference_system and self.reference_system.is_calibrated:
            is_above, confidence, ref_y = self.reference_system.check_ball_height_by_perspective(
                self.current_depth, ball_bottom_y
            )
            self.current_is_above = is_above if is_above is not None else False
            self.current_confidence = confidence
            
            # Record history
            self.depth_history.append((current_time, self.current_depth))
            self.y_history.append((current_time, ball_bottom_y, ref_y))
            
            # Update state machine
            self._update_state(current_time)
    
    def _update_state(self, current_time):
        """Update state machine"""
        if len(self.y_history) < 3:
            return
        
        # Analyze Y coordinate change trend
        recent_y = [y[1] for y in list(self.y_history)[-5:]]
        
        if len(recent_y) >= 2:
            # Calculate Y coordinate changes
            y_changes = []
            for i in range(1, len(recent_y)):
                y_changes.append(recent_y[i] - recent_y[i-1])
            
            avg_change = sum(y_changes) / len(y_changes) if y_changes else 0
            
            # State transitions
            if self.current_state == BallState.INITIAL:
                # Ball starts rising (Y coordinate decreases)
                if avg_change < -2:  # Pixel change threshold
                    self._transition_to(BallState.RISING)
                    self.just_reached_peak = False
                    
            elif self.current_state == BallState.RISING:
                # Ball starts falling (Y coordinate increases)
                if avg_change > 2:
                    self._transition_to(BallState.FALLING)
                    self.just_reached_peak = True
                    
                    # Check if pass is valid
                    if self.current_confidence > 0.7 and self.current_is_above:
                        self.count += 1
                        print(f"\n排球计数 +1，总得分: {self.count}")
                        print(f"深度: {self.current_depth:.1f}m, 最高点Y轴: {self.y_history[-1][1]:.1f}m, 参考系Y轴: {self.reference_system.get_y_axis_height(self.current_depth):.1f}m")
                    else:
                        print(f"\n高度不足！深度:{self.current_depth:.1f}m, 最高点Y轴: {self.y_history[-1][1]:.1f}m, 参考系Y轴: {self.reference_system.get_y_axis_height(self.current_depth):.1f}m")

            elif self.current_state == BallState.FALLING:
                # Ball rises again
                if avg_change < -2:
                    self._transition_to(BallState.RISING)
                    self.just_reached_peak = False
    
    def _transition_to(self, new_state):
        """State transition"""
        self.current_state = new_state
    
    def start_exam(self):
        """Start exam"""
        self.exam_state = ExamState.RUNNING
        self.exam_start_time = time()
        self.count = 0
        self.depth_history.clear()
        self.y_history.clear()
        self.current_state = BallState.INITIAL
        print(f"\n开始考试！时长: {EXAM_DURATION} 秒")

    def update_exam_time(self):
        """Update exam time"""
        if self.exam_state == ExamState.RUNNING:
            elapsed = time() - self.exam_start_time
            self.exam_remaining_time = max(0, EXAM_DURATION - elapsed)
            
            if self.exam_remaining_time <= 0:
                self.finish_exam()
    
    def finish_exam(self):
        """Finish exam"""
        self.exam_state = ExamState.FINISHED
        print(f"\n考试结束！最终得分: {self.count} 个")
    
    def get_status_text(self):
        """Get status text"""
        status_lines = []
        status_lines.append(f"Status: {self.exam_state.value}")
        
        if self.exam_state == ExamState.RUNNING:
            status_lines.append(f"Countdown: {int(self.exam_remaining_time)} seconds")
            status_lines.append(f"Volleyball Count: {self.count} pieces")
            status_lines.append(f"Current State: {self.current_state.value}")
            status_lines.append(f"Depth: {self.current_depth:.2f}m")

            if self.reference_system and self.reference_system.is_calibrated:
                status_lines.append(f"Height Assessment: {'Qualified' if self.current_is_above else 'Not Qualified'} (Confidence: {self.current_confidence:.2f})")

        elif self.exam_state == ExamState.FINISHED:
            status_lines.append(f"Finally Score: {self.count} pieces")

        return status_lines

# Global counter instances
volleyball_counter = None
reference_system = None

def xywh2abcd(xywh, im_shape):
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
    """Optimized YOLO inference thread"""
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
        if "connection" in str(e).lower() or "network" in str(e).lower():
            print(f"Network error while loading model: {e}")
            print(f"Model file may not exist at: {weights}")
            print("Please check the model file path")
        else:
            print(f"Error loading model: {e}")
        exit_signal = True
        return
    
    inference_fps_counter = FPSCounter(window_size=30)
    
    while not exit_signal:
        try:
            image_data = image_queue.get(timeout=0.1)
            img_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
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
            print(f"Inference Error: {e}")

def render_2D_perspective(image, image_scale, objects):
    """Perspective-based 2D rendering"""
    global volleyball_counter, reference_system
    
    # Draw reference lines first (if calibrated) - 6 points connection
    if reference_system and reference_system.is_calibrated:
        reference_system.visualize_reference(image)

    # Process currently detected objects
    ball_depth = None
    ball_y = None
    ref_y = None
    
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # Get bounding box
            bbox = obj.bounding_box_2d
            
            # Calculate ball bottom center Y coordinate
            bottom_center_x = (bbox[2][0] + bbox[3][0]) / 2
            bottom_center_y = (bbox[2][1] + bbox[3][1]) / 2  # Bottom Y coordinate
            
            # Scale to display coordinates
            display_bottom_x = int(bottom_center_x * image_scale[0])
            display_bottom_y = int(bottom_center_y * image_scale[1])
            
            # Update counter
            if volleyball_counter:
                volleyball_counter.update(obj.position, bottom_center_y)
                ball_depth = obj.position[2]
                ball_y = bottom_center_y
                
                # Get reference Y coordinate
                if reference_system and reference_system.is_calibrated:
                    _, _, ref_y_temp = reference_system.check_ball_height_by_perspective(
                        ball_depth, bottom_center_y
                    )
                    ref_y = ref_y_temp
            
            # Draw bounding box
            color = (0, 255, 0) if volleyball_counter and volleyball_counter.just_reached_peak else (0, 200, 0)
            thickness = 3 if volleyball_counter and volleyball_counter.just_reached_peak else 2
            
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         color, thickness)
            
            # Draw bottom center point
            cv2.circle(image, (display_bottom_x, display_bottom_y), 5, (255, 0, 0), -1)
            
            # Display depth information
            distance = np.linalg.norm(obj.position) if obj.position[0] != 0 else 0
            label_text = f"D={distance:.2f}m"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw reference lines and judgment results (with 6 points connection)
    if reference_system and reference_system.is_calibrated and ball_depth and ball_y:
        reference_system.visualize_reference(image, ball_depth, ball_y, ref_y)
    
    # Display exam status information with better visibility
    if volleyball_counter:
        volleyball_counter.update_exam_time()
        
        # Draw semi-transparent background for better text visibility
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Display status text
        status_lines = volleyball_counter.get_status_text()
        y_offset = 30
        for line in status_lines:
            # Choose color based on content
            if "Qualified" in line or "Success" in line:
                color = (0, 255, 255)  # Yellow for success
            elif "Not Qualified" in line:
                color = (0, 0, 255)  # Red for failure
            else:
                color = (255, 255, 255)  # White for normal text
            
            cv2.putText(image, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Display FPS information at bottom
        cv2.putText(image, f"Camera FPS: {camera_fps:.1f} | Inference FPS: {inference_fps:.1f}",
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Global FPS variable
camera_fps = 0.0

def main():
    global exit_signal, volleyball_counter, camera_fps, inference_fps, reference_system
    
    # Check model file before starting
    if not os.path.exists(opt.weights):
        print(f"Error: Model file not found at {opt.weights}")
        print("Please specify correct model path using --weights argument")
        return
    
    # Create perspective reference system
    reference_system = PrecisePerspectiveSystem()
    
    # Load calibration data - exit if failed
    if not reference_system.load_calibration():
        print("\nError: Calibration data not found!")
        print("Please run calibration script first to generate perspective_calibration.json")
        return
    
    # Create counter
    volleyball_counter = PerspectiveVolleyballCounter(reference_system)
    
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
    init_params.camera_resolution = sl.RESOLUTION.HD1080
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
    
    # Enable position tracking - set as static camera
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.enable_pose_smoothing = True
    positional_tracking_parameters.set_as_static = True  # Set as static camera
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
    
    # Use lower display resolution for better performance
    display_resolution = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    
    # Create display image
    image_left_ocv = np.zeros((display_resolution.height, display_resolution.width, 4), np.uint8)
    image_left = sl.Mat()
    
    # FPS counter
    fps_counter = FPSCounter(window_size=30)
    
    print("\n排球垫球计数系统 (精确透视版)")
    print("操作说明:")
    print("- 按 'Q' 开始/结束考试")
    print("- 按 'ESC' 退出程序")
    print(f"\n标准杆高度: 2.35m\n")
    
    # Create window
    cv2.namedWindow("Volleyball Counter - Perspective", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Volleyball Counter - Perspective", display_resolution.width, display_resolution.height)
    
    # Flag variables
    detections = []
    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Update capture FPS
                camera_fps = fps_counter.update()

                # Get image for inference1
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
                
                # If there are detection results, inject to ZED
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
                    
                    # Render objects and information
                    render_2D_perspective(image_left_ocv, image_scale, objects)
                    
                    # Display image
                    cv2.imshow("Volleyball Counter - Perspective", image_left_ocv)
                
                # Keyboard control
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    exit_signal = True
                elif key == ord('q') or key == ord('Q'):
                    if volleyball_counter.exam_state == ExamState.IDLE or volleyball_counter.exam_state == ExamState.FINISHED:
                        volleyball_counter.start_exam()
                    elif volleyball_counter.exam_state == ExamState.RUNNING:
                        volleyball_counter.finish_exam()
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
    parser.add_argument('--weights', type=str, default='/usr/local/zed/volleyball.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()