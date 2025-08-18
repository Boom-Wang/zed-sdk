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

# 使用队列代替锁，提高并发性能
image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=2)
exit_signal = False

# 添加推理FPS的全局变量
inference_fps = 0.0

# 考试相关参数
class ExamState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"

class Gender(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"

# 排球考试参数
EXAM_DURATION = 41
HEIGHT_THRESHOLD_MALE = 2.23  # 男生垫球高度阈值（米）
HEIGHT_THRESHOLD_FEMALE = 2.15  # 女生垫球高度阈值（米）
INITIAL_HEIGHT = 1.2  # 初始状态高度阈值（米）

# FPS 计算优化
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

# 垫球状态机
class BallState(Enum):
    INITIAL = "INITIAL"
    RISING = "RISING"
    FALLING = "FALLING"

class VolleyballCounter:
    def __init__(self, camera_height, gender=Gender.MALE, use_bottom_detection=True):
        self.camera_height = camera_height
        self.gender = gender
        self.height_threshold = HEIGHT_THRESHOLD_MALE if gender == Gender.MALE else HEIGHT_THRESHOLD_FEMALE
        self.use_bottom_detection = use_bottom_detection
        self.volleyball_radius = 0.12  # 排球半径

        # 计数相关
        self.count = 0
        self.current_state = BallState.INITIAL
        self.last_state = BallState.INITIAL
        
        # 高度追踪
        self.max_height_in_cycle = 0
        self.min_height_in_cycle = float('inf')
        self.current_height = 0
        
        # 考试状态
        self.exam_state = ExamState.IDLE
        self.exam_start_time = 0
        self.exam_remaining_time = EXAM_DURATION
        
        # 高度历史用于判断上升/下降
        self.height_history = deque(maxlen=10)
        
        # 标记是否刚达到最高点
        self.just_reached_peak = False
        
    def update(self, ball_object):
        """更新排球状态和计数"""
        if self.exam_state != ExamState.RUNNING:
            return
        current_time = time()
        # self.current_height = self.camera_height + ball_position[1]
        if self.use_bottom_detection:
            if hasattr(ball_object, 'bounding_box_3d') and len(ball_object.bounding_box) > 0:
                bottom_points = ball_object.bounding_box[4:8]  # 取底部四个点
                bottom_y = min([point[1] for point in bottom_points])
                self.current_height = self.camera_height + bottom_y
            else:
                self.current_height = self.camera_height + ball_object.position[1] - self.volleyball_radius
        else:
            self.current_height = self.camera_height + ball_object.position[1]

        # 记录高度历史
        self.height_history.append((current_time, self.current_height))
        # 状态机逻辑
        self._update_state(current_time)
        # 更新最高/最低高度
        if self.current_state == BallState.RISING:
            self.max_height_in_cycle = max(self.max_height_in_cycle, self.current_height)
        elif self.current_state == BallState.FALLING:
            self.min_height_in_cycle = min(self.min_height_in_cycle, self.current_height)
    
    def _update_state(self, current_time):
        """更新状态机，使用高度变化趋势判断上升/下降"""
        # 需要足够的历史数据才能判断趋势
        if len(self.height_history) < 3:
            return
        
        # 计算最近的高度变化趋势
        recent_heights = [h[1] for h in list(self.height_history)[-5:]]
        # 使用简单的差分来判断趋势
        if len(recent_heights) >= 2:
            # 计算平均高度变化
            height_changes = []
            for i in range(1, len(recent_heights)):
                height_changes.append(recent_heights[i] - recent_heights[i-1])
            
            avg_change = sum(height_changes) / len(height_changes) if height_changes else 0
            
            # 状态转换逻辑
            if self.current_state == BallState.INITIAL:
                # 从初始状态开始上升
                if self.current_height > INITIAL_HEIGHT and avg_change > 0.02:
                    self._transition_to(BallState.RISING, current_time)
                    self.max_height_in_cycle = self.current_height
                    self.just_reached_peak = False
                    
            elif self.current_state == BallState.RISING:
                # 从上升转为下降（检测到最高点）
                if avg_change < -0.02:
                    self._transition_to(BallState.FALLING, current_time)
                    self.min_height_in_cycle = self.current_height
                    self.just_reached_peak = True  # 标记刚达到最高点
                    
                    # 判断是否达标并计数
                    if self.max_height_in_cycle >= self.height_threshold:
                        self.count += 1
                        print(f"✓ Pass Success! Max Height: {self.max_height_in_cycle:.2f}m, Count: {self.count}")
                    else:
                        print(f"✗ Height Insufficient! Max Height: {self.max_height_in_cycle:.2f}m < {self.height_threshold}m")
                    
            elif self.current_state == BallState.FALLING:
                # 从下降转为上升（新的周期开始）
                if avg_change > 0.02:
                    self._transition_to(BallState.RISING, current_time)
                    self.max_height_in_cycle = self.current_height
                    self.just_reached_peak = False
    
    def _transition_to(self, new_state, current_time):
        """状态转换"""
        self.last_state = self.current_state
        self.current_state = new_state
    
    def start_exam(self):
        """开始考试"""
        self.exam_state = ExamState.RUNNING
        self.exam_start_time = time()
        self.count = 0
        self.height_history.clear()
        self.current_state = BallState.INITIAL
        print(f"Exam Started! Duration: {EXAM_DURATION} seconds")
    
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
        print(f"\nExam Finished! Final Score: {self.count} passes")
    
    def get_status_text(self):
        """获取状态文本"""
        status_lines = []
        status_lines.append(f"Gender: {self.gender.value} | Threshold: {self.height_threshold}m")
        status_lines.append(f"Status: {self.exam_state.value}")
        
        if self.exam_state == ExamState.RUNNING:
            status_lines.append(f"Time: {int(self.exam_remaining_time)}s")
            status_lines.append(f"Count: {self.count}")
            status_lines.append(f"State: {self.current_state.value}")
            status_lines.append(f"Height: {self.current_height:.2f}m")
            
            if self.current_state == BallState.RISING or self.current_state == BallState.FALLING:
                status_lines.append(f"Max in cycle: {self.max_height_in_cycle:.2f}m")
                
        elif self.exam_state == ExamState.FINISHED:
            status_lines.append(f"Final Score: {self.count} passes")
            
        return status_lines

# 全局计数器实例
volleyball_counter = None

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
    """优化的YOLO推理线程"""
    global exit_signal, inference_fps
    print("Initializing Network...")
    model = YOLO(weights)
    
    # 预热模型
    dummy_input = np.zeros((720, 1280, 4), dtype=np.uint8)
    dummy_rgb = cv2.cvtColor(dummy_input, cv2.COLOR_RGBA2RGB)
    _ = model.predict(dummy_rgb, save=False, imgsz=img_size, verbose=False)
    print("Model warmed up")
    
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

def render_2D_simple(image, image_scale, objects):
    """简化的2D渲染，只显示当前检测框和信息"""
    global volleyball_counter
    
    # 处理当前检测到的对象
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # 更新计数器
            if volleyball_counter:
                volleyball_counter.update(obj)
            
            # 获取边界框
            bbox = obj.bounding_box_2d

            width = bbox[2][0] - bbox[0][0]
            height = bbox[2][1] - bbox[0][1]
            area = width * height

            # 绘制边界框
            color = (0, 255, 0) if volleyball_counter and volleyball_counter.just_reached_peak else (0, 200, 0)
            thickness = 3 if volleyball_counter and volleyball_counter.just_reached_peak else 2
            
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         color, thickness)
            
            # 显示信息
            position = obj.position
            ball_height = volleyball_counter.camera_height + position[1] if volleyball_counter else 0
            distance = np.linalg.norm(obj.position) if obj.position[0] != 0 else 0
            label_text = f"D={distance:.2f}m  Area={area:.0f} Size={width:.0f}x{height:.0f}"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 显示考试状态信息
    if volleyball_counter:
        volleyball_counter.update_exam_time()
        status_lines = volleyball_counter.get_status_text()
        
        y_offset = 30
        for line in status_lines:
            color = (0, 255, 255) if "Pass Success" in line else (255, 255, 0)
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # 显示FPS信息
        cv2.putText(image, f"Camera FPS: {camera_fps:.1f} | Inference FPS: {inference_fps:.1f}",
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# 全局FPS变量
camera_fps = 0.0

def main():
    global exit_signal, volleyball_counter, camera_fps, inference_fps
    
    # 初始化相机高度和性别
    camera_height = 1.60
    gender = Gender.MALE  # 可以通过参数设置
    
    volleyball_counter = VolleyballCounter(camera_height, gender)
    
    capture_thread = Thread(target=torch_thread, kwargs={
        'weights': opt.weights, 
        'img_size': opt.img_size, 
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres
    })
    capture_thread.start()
    print("Initializing Camera...")
    zed = sl.Camera()
    
    # 初始化参数
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10
    init_params.depth_minimum_distance = 2.5
    init_params.depth_stabilization = 10
    
    # 处理输入源
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
    runtime_params.confidence_threshold = 50
    runtime_params.texture_confidence_threshold = 50
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()
    
    image_left_tmp = sl.Mat()
    print("Initialized Camera")
    
    # 启用位置追踪 - 设置为静态相机，以地面为原点
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True  # 关闭区域记忆以提高性能
    positional_tracking_parameters.enable_pose_smoothing = True
    positional_tracking_parameters.set_as_static = True  # 设置为静态相机
    # positional_tracking_parameters.set_floor_as_origin = True  # 以地面为原点
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # 配置物体检测
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    obj_param.max_range = 10  # 减少检测范围以提高性能
    
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Enable object detection failed: {repr(err)}. Exit program.")
        zed.close()
        exit()
    
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # 显示设置
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    
    # 使用较低的显示分辨率以提高性能
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    
    # 创建显示图像
    image_left_ocv = np.zeros((display_resolution.height, display_resolution.width, 4), np.uint8)
    image_left = sl.Mat()
    
    # FPS计数器
    fps_counter = FPSCounter(window_size=30)
    
    # 控制渲染频率的变量
    last_render_time = 0
    render_interval = 0.1 # 最小渲染间隔（秒）
    should_render = False
    
    print("\nVolleyball Exam Counter System (Optimized)")
    print("Instructions:")
    print("- Press 'S' to start/stop exam")
    print("- Press 'ESC' to exit")
    print("- Press 'G' to switch gender")
    print(f"\nCurrent Settings: {gender.value}, Height Threshold: {volleyball_counter.height_threshold}m\n")
    
    # 创建窗口
    cv2.namedWindow("Volleyball Exam Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Volleyball Exam Counter", display_resolution.width, display_resolution.height)
    
    # 标志变量
    detections = []
    
    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 更新采集FPS
                camera_fps = fps_counter.update()
                current_time = time()
                
                # 获取图像用于推理
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                
                # 复制图像数据到队列
                if image_net is not None and image_net.size > 0:
                    try:
                        if image_queue.full():
                            image_queue.get_nowait()
                        image_queue.put(image_net.copy())
                    except:
                        pass
                
                # 获取检测结果（非阻塞）
                try:
                    detections = detection_queue.get_nowait()
                except Empty:
                    pass
                
                # 如果有检测结果，注入到ZED
                if detections and len(detections) > 0:
                    zed.ingest_custom_box_objects(detections)
                
                # 获取跟踪的对象
                zed.retrieve_objects(objects, obj_runtime_param)
                
                # 判断是否需要渲染
                if volleyball_counter:
                    if volleyball_counter.just_reached_peak:
                        should_render = True
                        volleyball_counter.just_reached_peak = False  # 重置标志
                    elif current_time - last_render_time > render_interval:
                        should_render = True
                else:
                    # 如果没有计数器，定期渲染
                    if current_time - last_render_time > render_interval:
                        should_render = True
                
                # 只在需要时渲染和显示
                if should_render:
                    # 获取显示图像
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    
                    # 将图像数据复制到OpenCV格式
                    image_data_gpu = image_left.get_data()
                    if image_data_gpu is not None:
                        np.copyto(image_left_ocv, image_data_gpu)
                        
                        # 渲染对象和信息
                        render_2D_simple(image_left_ocv, image_scale, objects)
                        
                        # 显示图像
                        cv2.imshow("Volleyball Exam Counter", image_left_ocv)
                        
                        last_render_time = current_time
                        should_render = False
                
                # 键盘控制（每帧都检查）
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    exit_signal = True
                elif key == ord('s') or key == ord('S'):
                    if volleyball_counter.exam_state == ExamState.IDLE or volleyball_counter.exam_state == ExamState.FINISHED:
                        volleyball_counter.start_exam()
                        should_render = True
                    elif volleyball_counter.exam_state == ExamState.RUNNING:
                        volleyball_counter.finish_exam()
                        should_render = True
                elif key == ord('g') or key == ord('G'):
                    # 切换性别
                    new_gender = Gender.FEMALE if volleyball_counter.gender == Gender.MALE else Gender.MALE
                    volleyball_counter = VolleyballCounter(camera_height, new_gender)
                    print(f"Gender switched to: {new_gender.value}, Height Threshold: {volleyball_counter.height_threshold}m")
                    should_render = True
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
    parser.add_argument('--weights', type=str, default='/home/laplace/data/zed-env/zed-sdk/volleyball_0801.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()