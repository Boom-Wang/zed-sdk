#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep, time
import cv_viewer.tracking_viewer as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False
detections = []

# 全局配置
USE_BOTTOM_DETECTION = True  # 是否使用底部检测
VOLLEYBALL_RADIUS = 0.12  # 标准排球半径（米）

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
            obj.label = det.cls[0]
            obj.probability = det.conf[0]
            obj.is_grounded = False
            output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, model
    print("Intializing Network...")
    model = YOLO(weights)
    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            det = model.predict(img, save=False, imgsz=img_size, verbose=False, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
            detections = detections_to_custom_box(det, image_net, model)
            lock.release()
            run_signal = False
        sleep(0.01)

def get_object_height(obj, camera_height, use_bottom=True):
    position = obj.position
    
    if use_bottom:
        # 尝试使用3D边界框
        if hasattr(obj, 'bounding_box') and len(obj.bounding_box) > 0:
            # 3D边界框通常包含8个顶点，底部4个顶点通常是后4个或Y值最小的
            try:
                bottom_y = min([vertex[1] for vertex in obj.bounding_box])
                height = camera_height + bottom_y
                mode = "3D_BOX"
            except:
                height = camera_height + position[1] - VOLLEYBALL_RADIUS
                mode = "CENTER-R"
        else:
            # 使用中心点减去半径
            height = camera_height + position[1] - VOLLEYBALL_RADIUS
            mode = "CENTER-R"
    else:
        # 使用中心点高度
        height = camera_height + position[1]
        mode = "CENTER"
    return height, mode

def render_2D_with_height(image, image_scale, objects, camera_height, use_bottom_detection=True):
    """
    Args:
        image: 图像数组
        image_scale: 图像缩放比例
        objects: 检测到的物体
        camera_height: 相机高度
        use_bottom_detection: 是否使用底部检测
    """
    cv_viewer.render_2D(image, image_scale, objects, True)

    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # 计算高度
            height, detection_mode = get_object_height(obj, camera_height, use_bottom_detection)
            
            # 计算距离
            position = obj.position
            distance = np.linalg.norm(position)
            # 准备标签文本
            label_text = f"H: {height:.2f}m | D: {distance:.2f}m | Mode: {detection_mode}"
            # 获取边界框位置
            bbox = obj.bounding_box_2d
            bbox_top_left = (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]))
            
            cv2.putText(image, label_text,
                       (bbox_top_left[0], bbox_top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            

def main():
    global image_net, exit_signal, run_signal, detections, USE_BOTTOM_DETECTION
    frame_count = 0
    fps_timer = cv2.getTickCount()
    Camera_height = 1.50
    
    # 从命令行参数获取检测模式
    USE_BOTTOM_DETECTION = opt.use_bottom
    
    detection_mode_str = "底部检测" if USE_BOTTOM_DETECTION else "中心检测"
    print(f"相机高度已设置为: {Camera_height} 米")
    print(f"检测模式: {detection_mode_str}")
    print(f"排球半径设置: {VOLLEYBALL_RADIUS} 米")

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_fps = 30
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 20.0
    init_params.depth_minimum_distance = 0.3
    init_params.svo_real_time_mode = False
    init_params.depth_stabilization = 30

    if opt.ip is not None:
        print(f"Connecting to remote stream at {opt.ip}")
        try:
            ip_address, port = opt.ip.split(':')
            port = int(port)
            init_params.set_from_stream(ip_address, port)
        except ValueError:
            print(f"Invalid IP format. Please use format: IP:PORT (e.g., 192.168.8.14:30000)")
            exit()
    elif opt.svo is not None:
        input_type = sl.InputType()
        input_type.set_from_svo_file(opt.svo)
        init_params.input = input_type

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()

    image_left_tmp = sl.Mat()
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    positional_tracking_parameters.enable_pose_smoothing = True
    positional_tracking_parameters.enable_area_memory = True
    # positional_tracking_parameters.set_floor_as_origin = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    obj_param.max_range = 10
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
    image_left = sl.Mat()

    print("\n控制说明:")
    print("- 按 'Q' 或 'ESC' 退出程序")
    print(f"- 当前模式: {detection_mode_str}\n")

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            current_time = time()
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            while run_signal:
                sleep(0.001)

            lock.acquire()
            if len(detections) > 0:
                zed.ingest_custom_box_objects(detections)
                zed.retrieve_objects(objects, obj_runtime_param) 

                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
                render_2D_with_height(image_left_ocv, image_scale, objects, Camera_height, USE_BOTTOM_DETECTION)
            else:
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
            
            lock.release()

            if frame_count % 30 == 0:
                current_time = cv2.getTickCount()
                fps = 30 / ((current_time - fps_timer) / cv2.getTickFrequency())
                fps_timer = current_time
            
            cv2.putText(image_left_ocv, f"FPS: {fps:.2f}" if 'fps' in locals() else "FPS: --",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            mode_text = f"Mode: {'Bottom' if USE_BOTTOM_DETECTION else 'Center'} | R: {VOLLEYBALL_RADIUS:.2f}m"
            cv2.putText(image_left_ocv, mode_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2) 
            cv2.imshow("ZED | 2D Object Detection with Height", image_left_ocv)
            
            # 键盘控制
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
 
        else:
            exit_signal = True
    
    exit_signal = True
    capture_thread.join()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/laplace/data/zed-env/zed-sdk/volleyball.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming (e.g., 192.168.8.14:30000)')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--use_bottom', action='store_true', help='use bottom detection instead of center detection')
    opt = parser.parse_args()

    with torch.no_grad():
        main()