#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

lock = Lock()
run_signal = False
exit_signal = False


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2])
    x_max = (xywh[0] + 0.5*xywh[2])
    y_min = (xywh[1] - 0.5*xywh[3])
    y_max = (xywh[1] + 0.5*xywh[3])

    # A ------ B
    # | Object |
    # D ------ C

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
        
        # 获取类别标签
        cls_id = int(det.cls[0])
        label = model.names[cls_id]
        
        # 只处理人类检测
        if label == "person":
            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
            obj.label = det.cls
            obj.probability = det.conf
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
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net, model)
            lock.release()
            run_signal = False
        sleep(0.01)


def render_2D_with_distance(image, image_scale, objects):
    """自定义的2D渲染函数，显示检测框和距离"""
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # 绘制边界框
            bbox = obj.bounding_box_2d
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         (0, 255, 0), 2)
            
            # 计算距离并显示
            distance = np.linalg.norm(obj.position)
            label_text = f"Person {obj.id}: {distance:.2f}m"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 40
    init_params.depth_minimum_distance = 0.5
    init_params.svo_real_time_mode = True
    init_params.depth_stabilization = 100

    # 处理输入源
    if opt.ip is not None:
        # 使用远程IP流
        print(f"Connecting to remote stream at {opt.ip}")
        try:
            ip_address, port = opt.ip.split(':')
            port = int(port)
            init_params.set_from_stream(ip_address, port)
        except ValueError:
            print(f"Invalid IP format. Please use format: IP:PORT (e.g., 192.168.8.14:30000)")
            exit()
    elif opt.svo is not None:
        # 使用SVO文件
        input_type = sl.InputType()
        input_type.set_from_svo_file(opt.svo)
        init_params.input = input_type

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    
    # 只保留2D显示相关的设置
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
    image_left = sl.Mat()

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # 只有当检测到人时才进行后续处理
            if len(detections) > 0:
                # -- Ingest detections
                zed.ingest_custom_box_objects(detections)
                lock.release()
                zed.retrieve_custom_objects(objects, obj_runtime_param)

                # 只进行2D渲染
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
                render_2D_with_distance(image_left_ocv, image_scale, objects)
            else:
                lock.release()
                # 没有检测到人时，只显示原始图像
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
            
            # 只显示2D视图
            cv2.imshow("ZED | 2D Object Detection", image_left_ocv)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
        else:
            exit_signal = True

    exit_signal = True
    zed.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/laplace/data/yolov8/yolo/ultralytics/yolo11n.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming (e.g., 192.168.8.14:30000)')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()