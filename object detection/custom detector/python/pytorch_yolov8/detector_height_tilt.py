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
from time import sleep, time
from collections import deque
import cv_viewer.tracking_viewer as cv_viewer

# 使用队列代替锁，提高并发性能
image_queue = Queue(maxsize=6)
detection_queue = Queue(maxsize=4)
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

# def calculate_height_with_tilt(position, camera_height, pitch_angle):
#     """
#     根据相机倾斜角度计算物体真实高度
    
#     参数:
#     - position: 物体在相机坐标系中的位置 [x, y, z]
#     - camera_height: 相机距离地面的高度
#     - pitch_angle: 相机向下的倾斜角度（弧度）
    
#     返回:
#     - real_height: 物体的真实高度
#     """
#     if abs(pitch_angle) < 0.01:  # 如果倾斜角度接近0（水平）
#         return camera_height + position[1]
    
#     # 当相机有倾斜角度时的完整公式
#     # 将相机坐标系中的位置转换到世界坐标系
#     x_cam, y_cam, z_cam = position[0], position[1], position[2]
    
#     # 旋转矩阵（只考虑pitch角度）
#     cos_pitch = np.cos(pitch_angle)
#     sin_pitch = np.sin(pitch_angle)
    
#     # 转换到世界坐标系
#     y_world = y_cam * cos_pitch - z_cam * sin_pitch
    
#     # 真实高度 = 相机高度 + 世界坐标系中的y值
#     real_height = camera_height + y_world
    
#     return real_height

# def render_2D_with_height(image, image_scale, objects, camera_height, pitch_angle, capture_fps, inference_fps):
def render_2D_with_height(image, image_scale, objects, camera_height, capture_fps, inference_fps):
    """渲染2D图像并显示高度信息"""
    cv_viewer.render_2D(image, image_scale, objects, True)
    
    # 在图像顶部显示双FPS信息
    fps_text = f"Camera FPS: {capture_fps:.1f} | Inference FPS: {inference_fps:.1f}"
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示相机模式
    # mode_text = "Mode: Tilted" if abs(pitch_angle) >= 0.01 else "Mode: Horizontal"
    # cv2.putText(image, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 显示倾斜角度
    # angle_deg = np.degrees(pitch_angle)
    # angle_text = f"Pitch Angle: {angle_deg:.1f}"
    # cv2.putText(image, angle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            position = obj.position
            
            # 计算高度
            H_height = camera_height + position[1]
            # 根据倾斜角度计算高度
            # H_height = calculate_height_with_tilt(position, camera_height, pitch_angle)
            
            distance = np.linalg.norm(position)
            label_text = f"Height: {H_height:.2f}m | Dist: {distance:.2f}m"
            
            bbox_top_left = (int(obj.bounding_box_2d[0][0] * image_scale[0]), 
                           int(obj.bounding_box_2d[0][1] * image_scale[1]))
            cv2.putText(image, label_text,
                       (bbox_top_left[0], bbox_top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

def main():
    global exit_signal, inference_fps
    Camera_height = 1.45
    print(f"相机高度已设置为: {Camera_height} 米")
    
    # 启动推理线程
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
    init_params.depth_maximum_distance = 15
    init_params.depth_minimum_distance = 0.5
    init_params.svo_real_time_mode = False
    init_params.depth_stabilization = 20

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

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 10
    runtime_params.texture_confidence_threshold = 10

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()

    image_left_tmp = sl.Mat()
    print("Initialized Camera")

    # 启用位置跟踪（获取相机姿态信息）
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True
    positional_tracking_parameters.set_as_static = True  # 设置为静态模式
    # positional_tracking_parameters.enable_imu_fusion = True  # 启用IMU融合以获得更准确的姿态
    zed.enable_positional_tracking(positional_tracking_parameters)

    # 配置物体检测
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    # 用于存储相机姿态
    # camera_pose = sl.Pose()

    # 显示设置
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
    image_left = sl.Mat()

    # FPS计数器
    capture_fps_counter = FPSCounter(window_size=30)
    last_print_time = time()
    print_interval = 0.1  # 100ms打印间隔

    print("\nStarting Detection... (Press 'Ctrl+C' to exit)\n")

    try:
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 更新采集FPS
                capture_fps = capture_fps_counter.update()
                current_time = time()
                
                # 获取相机姿态
                # tracking_state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                
                # 获取倾斜角度（pitch）
                # pitch_angle = 0.0
                # if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                #     rotation = camera_pose.get_rotation_matrix()
                #     # 从旋转矩阵提取pitch角度
                #     pitch_angle = np.arctan2(-rotation[2, 0], 
                #                            np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
                
                # 获取图像并放入队列
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_data = image_left_tmp.get_data().copy()
                
                try:
                    if image_queue.full():
                        image_queue.get_nowait()
                    image_queue.put(image_data)
                except:
                    pass
                
                # 尝试获取检测结果
                try:
                    detections = detection_queue.get_nowait()
                    
                    if len(detections) > 0:
                        zed.ingest_custom_box_objects(detections)
                        zed.retrieve_objects(objects, obj_runtime_param)
                        
                        # 渲染图像
                        zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                        np.copyto(image_left_ocv, image_left.get_data())
                        # render_2D_with_height(image_left_ocv, image_scale, objects, 
                        #                     Camera_height, pitch_angle, capture_fps, inference_fps)
                        render_2D_with_height(image_left_ocv, image_scale, objects, 
                                            Camera_height, capture_fps, inference_fps)
                    else:
                        # 没有检测到物体
                        zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                        np.copyto(image_left_ocv, image_left.get_data())
                        
                        # 显示FPS和状态信息
                        fps_text = f"Camera FPS: {capture_fps:.1f} | Inference FPS: {inference_fps:.1f}"
                        cv2.putText(image_left_ocv, fps_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # mode_text = "Mode: Tilted" if abs(pitch_angle) >= 0.01 else "Mode: Horizontal"
                        # cv2.putText(image_left_ocv, mode_text, (10, 60), 
                        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # angle_deg = np.degrees(pitch_angle)
                        # angle_text = f"Pitch Angle: {angle_deg:.1f}°"
                        # cv2.putText(image_left_ocv, angle_text, (10, 90), 
                        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        cv2.putText(image_left_ocv, "No Volleyball Detected", (10, 120), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                except Empty:
                    # 还没有检测结果
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    np.copyto(image_left_ocv, image_left.get_data())
                    
                    # 显示基本信息
                    fps_text = f"Camera FPS: {capture_fps:.1f} | Inference FPS: {inference_fps:.1f}"
                    cv2.putText(image_left_ocv, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow("ZED | 2D Object Detection with Height", image_left_ocv)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q') or key == ord('Q'):
                    exit_signal = True
                    
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
    capture_thread.join(timeout=2.0)
    zed.close()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/volleyball-int8-416.engine', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    opt = parser.parse_args()

    with torch.no_grad():
        main()