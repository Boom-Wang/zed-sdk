#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Thread, Event
from queue import Queue, Empty
from time import time
from collections import deque

# 使用队列代替锁，提高并发性能
image_queue = Queue(maxsize=6)  # 限制队列大小避免内存增长
detection_queue = Queue(maxsize=2)
exit_signal = False

# 添加推理FPS的全局变量
inference_fps = 0.0

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

def xywh2abcd(xywh, im_shape):
    """将YOLO格式的边界框转换为ZED格式"""
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
    """将YOLO检测结果转换为ZED自定义框格式"""
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        
        if label == "volleyball":
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
            obj.label = det.cls[0]
            obj.probability = det.conf[0]
            obj.is_grounded = False
            output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
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
            # 使用超时避免无限等待
            image_data = image_queue.get(timeout=0.1)
            # YOLO推理
            img_rgb = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
            results = model.predict(img_rgb, save=False, imgsz=img_size, verbose=False, conf=conf_thres, iou=iou_thres)[0]
            
            # 转换检测结果
            det_boxes = results.cpu().numpy().boxes
            detections = detections_to_custom_box(det_boxes, image_data, model)
            
            # 更新推理FPS
            inference_fps = inference_fps_counter.update()
            
            # 将结果放入队列
            try:
                # 如果队列满了，丢弃旧结果
                if detection_queue.full():
                    detection_queue.get_nowait()
                detection_queue.put(detections)
            except:
                pass
                
        except Empty:
            continue
        except Exception as e:
            print(f"Inference Error: {e}")

def print_objects_info(objects, camera_height, capture_fps, inference_fps, object_type="Volleyball"):
    """打印检测到的物体信息（包含双FPS显示）"""
    print(f"\033[H\033[J", end='')  # 清屏
    
    # 显示双FPS信息
    print(f"Camera FPS: {capture_fps:.1f} | Inference FPS: {inference_fps:.1f}")

    # 计算并显示处理比例
    if capture_fps > 0:
        process_ratio = inference_fps / capture_fps * 100
        print(f"Processing Rate: {process_ratio:.1f}% ({inference_fps:.1f}/{capture_fps:.1f})")

    print(f"\nCamera Height: {camera_height} m")
    print(f"Detected {object_type} Count: {len(objects.object_list)}")
    print("-" * 50)
    
    for i, obj in enumerate(objects.object_list):
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            position = obj.position
            H_height = camera_height + position[1]
            distance = np.linalg.norm(position)
            confidence = obj.confidence
            
            print(f"{object_type} {i+1}:")
            print(f"  Height: {H_height:.2f} m")
            print(f"  Distance: {distance:.2f} m")
            print(f"  Confidence: {confidence:.2f}")
            print("-" * 30)

def main():
    global exit_signal, inference_fps
    
    Camera_height = 0.78
    print(f"相机高度已设置为: {Camera_height} 米")
    
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
    init_params.camera_fps = 60
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10.0
    init_params.depth_minimum_distance = 0.5
    init_params.svo_real_time_mode = False
    init_params.depth_stabilization = 50
    init_params.enable_image_enhancement = False  # 关闭图像增强
    
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
    
    # 打开相机
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50  # 降低深度置信度阈值以提高性能
    runtime_params.texture_confidence_threshold = 50
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()
    
    print("Initialized Camera")
    
    # 启用位置跟踪
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = False  # 关闭区域记忆以提高性能
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # 配置物体检测
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False  # 确保关闭分割
    zed.enable_object_detection(obj_param)
    
    # 初始化对象
    image_left = sl.Mat()
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # FPS计数器
    fps_counter = FPSCounter(window_size=30)
    last_print_time = time()
    print_interval = 0.1  # 100ms打印间隔

    print("\nStarting Detection... (Press 'Ctrl+C' to exit)\n")

    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 更新采集FPS
                current_fps = fps_counter.update()
                current_time = time()
                
                # 获取图像并放入队列
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                image_data = image_left.get_data().copy()  # 复制数据避免覆盖
                
                try:
                    # 如果队列满了，丢弃旧图像
                    if image_queue.full():
                        image_queue.get_nowait()
                    image_queue.put(image_data)
                except:
                    pass
                
                # 尝试获取检测结果（非阻塞）
                try:
                    detections = detection_queue.get_nowait()
                    
                    if len(detections) > 0:
                        zed.ingest_custom_box_objects(detections)
                        zed.retrieve_objects(objects, obj_runtime_param)
                        
                        # 按指定间隔打印信息
                        if current_time - last_print_time >= print_interval:
                            print_objects_info(objects, Camera_height, current_fps, inference_fps, "排球")
                            last_print_time = current_time
                    else:
                        # 没有检测到物体时的输出
                        if current_time - last_print_time >= print_interval:
                            print(f"\033[H\033[J", end='')
                            print(f"Camera FPS: {current_fps:.1f} | Inference FPS: {inference_fps:.1f}")
                            if current_fps > 0:
                                process_ratio = inference_fps / current_fps * 100
                                print(f"Processing Rate: {process_ratio:.1f}% ({inference_fps:.1f}/{current_fps:.1f})")
                            print("\nNo Volleyball Detected")
                            last_print_time = current_time
                            
                except Empty:
                    # 还没有检测结果，继续
                    pass
                    
            else:
                print("Camera grab failed")
                exit_signal = True
                
    except KeyboardInterrupt:
        print("\n\nDetection stopped")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        exit_signal = True
    
    # 清理资源
    print("Cleaning up resources...")
    capture_thread.join(timeout=2.0)  # 等待推理线程结束
    zed.close()
    print("Program exited")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/volleyball-int8-416.engine', help='model path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()