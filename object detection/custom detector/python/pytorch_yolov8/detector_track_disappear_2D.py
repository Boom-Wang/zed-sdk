#!/usr/bin/env python3
import sys
import numpy as np
from collections import deque, defaultdict
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep, time

lock = Lock()
run_signal = False
exit_signal = False
camera_height = 1.45
object_tracks = defaultdict(lambda: deque(maxlen=300))  # 增加存储容量以支持渐隐效果

# 轨迹参数
TRACK_LIFETIME = 1.0  # 轨迹保持完全可见的时间（秒）
TRACK_FADEOUT_TIME = 0.5  # 轨迹渐隐消失的时间（秒）
TRACK_TOTAL_TIME = TRACK_LIFETIME + TRACK_FADEOUT_TIME 

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
    global image_net, exit_signal, run_signal, detections, model

    print("Intializing Network...")
    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            det = model(img, task='detect', save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net, model)
            lock.release()
            run_signal = False
        sleep(0.01)

def draw_gradient_line_with_fade(image, points, current_time, color_start=(0, 255, 0), color_end=(0, 100, 0), thickness=2):
    if len(points) < 2:
        return
    # 绘制每个线段
    for i in range(1, len(points)):
        # 获取两个点的位置和时间
        pt1_pos, pt1_time = points[i-1]
        pt2_pos, pt2_time = points[i]
        
        # 计算点的年龄（秒）
        age1 = current_time - pt1_time
        age2 = current_time - pt2_time
        
        # 如果两个点都太老了，跳过绘制
        # if age1 > TRACK_TOTAL_TIME and age2 > TRACK_TOTAL_TIME:
        #     continue
        
        # 计算平均年龄用于确定线段的透明度
        avg_age = (age1 + age2) / 2
        # 计算透明度（0-1）
        if avg_age < TRACK_LIFETIME:
            alpha = 1.0  # 完全不透明
        elif avg_age < TRACK_TOTAL_TIME:
            # 渐隐阶段
            alpha = 1.0 - (avg_age - TRACK_LIFETIME) / TRACK_FADEOUT_TIME
        else:
            continue
        
        # 根据位置计算颜色渐变
        position_alpha = i / len(points)
        base_color = (
            int(color_start[0] * (1 - position_alpha) + color_end[0] * position_alpha),
            int(color_start[1] * (1 - position_alpha) + color_end[1] * position_alpha),
            int(color_start[2] * (1 - position_alpha) + color_end[2] * position_alpha)
        )
        
        # 应用时间透明度到颜色
        final_color = (
            int(base_color[0] * alpha),
            int(base_color[1] * alpha),
            int(base_color[2] * alpha)
        )
        
        # 计算线条粗细（越新的轨迹越粗，同时考虑渐隐）
        line_thickness = max(1, int(thickness * (i / len(points)) * alpha))
        
        # 绘制线段
        cv2.line(image, 
                 (int(pt1_pos[0]), int(pt1_pos[1])), 
                 (int(pt2_pos[0]), int(pt2_pos[1])), 
                 final_color, line_thickness)

def clean_old_track_points(track, current_time):
    while track and (current_time - track[0][1]) > TRACK_TOTAL_TIME:
        track.popleft()

def render_2D_with_tracks(image, image_scale, objects, object_label="ball"):
    global object_tracks
    current_time = time()
    
    for track in object_tracks.values():
        clean_old_track_points(track, current_time)
    
    # 首先绘制所有轨迹
    for obj_id, track in object_tracks.items():
        if len(track) > 1:
            # 转换轨迹点到显示坐标，同时保留时间戳
            scaled_track = [((pt[0][0] * image_scale[0], pt[0][1] * image_scale[1]), pt[1]) 
                           for pt in track]
            # 绘制带渐隐效果的轨迹线
            draw_gradient_line_with_fade(image, scaled_track, current_time,
                                        color_start=(0, 50, 0),   # 深绿色（旧轨迹）
                                        color_end=(0, 255, 0),     # 亮绿色（新轨迹）
                                        thickness=3)
    
    # 然后绘制当前检测到的对象
    for obj in objects.object_list:
        if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            # 获取边界框中心点
            bbox = obj.bounding_box_2d
            center_x = (bbox[0][0] + bbox[2][0]) / 2
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            
            # 更新轨迹，添加时间戳
            object_tracks[obj.id].append(((center_x, center_y), current_time))
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1])),
                         (int(bbox[2][0] * image_scale[0]), int(bbox[2][1] * image_scale[1])),
                         (0, 255, 0), 2)
            
            # 计算距离并显示
            distance = np.linalg.norm(obj.position) if obj.position[0] != 0 else 0
            label_text = f"{object_label} {obj.id}: {distance:.2f}m"
            
            # 添加速度信息（如果可用）
            # if obj.velocity[0] != 0 or obj.velocity[1] != 0 or obj.velocity[2] != 0:
            #     speed = np.linalg.norm(obj.velocity)
            #     label_text += f" | {speed:.2f}m/s"
            position = obj.position
            H_height = camera_height + position[1]
            label_text += f" | {H_height:.2f}m"
            
            cv2.putText(image, label_text,
                       (int(bbox[0][0] * image_scale[0]), int(bbox[0][1] * image_scale[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    # 清理不再追踪的对象轨迹
    current_ids = [obj.id for obj in objects.object_list if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK]
    to_remove = []
    for track_id in object_tracks.keys():
        if track_id not in current_ids:
            # 检查轨迹是否已经完全过期
            if len(object_tracks[track_id]) == 0:
                to_remove.append(track_id)
            elif (current_time - object_tracks[track_id][-1][1]) > TRACK_TOTAL_TIME:
                # 如果最后一个点已经过期，删除整个轨迹
                to_remove.append(track_id)
    
    for track_id in to_remove:
        del object_tracks[track_id]

def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()
    print("Initializing Camera...")

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
    init_params.camera_fps = 120
    init_params.camera_resolution = sl.RESOLUTION.SVGA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 10
    init_params.depth_minimum_distance = 0.5
    init_params.depth_stabilization = 50

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
        init_params.svo_real_time_mode = False  # 实时播放SVO

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 40  # 深度置信度阈
    runtime_params.texture_confidence_threshold = 30

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open camera: {repr(status)}")
        exit()

    image_left_tmp = sl.Mat()
    print("Initialized Camera")

    # 启用位置追踪（即使只做2D追踪，这也能提供更好的追踪效果）
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_area_memory = True 
    positional_tracking_parameters.enable_pose_smoothing = True  # 平滑位姿
    zed.enable_positional_tracking(positional_tracking_parameters)

    # 配置物体检测参数
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True  # 启用追踪
    obj_param.enable_segmentation = False  # 不需要分割
    obj_param.max_range = 20 # 最大检测范围

    # 启用物体检测
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Enable object detection failed: {repr(err)}. Exit program.")
        zed.close()
        exit()

    objects = sl.Objects()
    
    # 配置运行时参数
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    obj_runtime_param.object_detection_properties.detection_confidence_threshold = 30
    # obj_runtime_param.object_detection_properties.tracking_max_dist = 5.0
    obj_runtime_param.object_detection_properties.tracking_timeout = 0.3
    
    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    
    # 只保留2D显示相关的设置
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
    image_left = sl.Mat()

    frame_count = 0
    fps_timer = cv2.getTickCount()

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            current_time = time()

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
            if len(detections) > 0:
                # -- Ingest detections
                zed.ingest_custom_box_objects(detections)
                lock.release()
                
                zed.retrieve_custom_objects(objects, obj_runtime_param)

                # 只进行2D渲染
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
                render_2D_with_tracks(image_left_ocv, image_scale, objects, "ball")
            else:
                lock.release()
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())
                
                for track in object_tracks.values():
                    clean_old_track_points(track, current_time)
                
                # 绘制现有轨迹
                for obj_id, track in object_tracks.items():
                    if len(track) > 1:
                        scaled_track = [((pt[0][0] * image_scale[0], pt[0][1] * image_scale[1]), pt[1]) 
                                       for pt in track]
                        draw_gradient_line_with_fade(image_left_ocv, scaled_track, current_time,
                                                    color_start=(0, 50, 0),
                                                    color_end=(0, 255, 0),
                                                    thickness=3)
            
            if frame_count % 30 == 0:
                current_time = cv2.getTickCount()
                fps = 30 / ((current_time - fps_timer) / cv2.getTickFrequency())
                fps_timer = current_time
                print(f"FPS: {fps:.2f}")
            
            cv2.putText(image_left_ocv, f"FPS: {fps:.2f}" if 'fps' in locals() else "FPS: --",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("ZED | 2D Object Detection with Tracking", image_left_ocv)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
            elif key == ord('c') or key == ord('C'):
                object_tracks.clear()
                print("Tracks cleared!")
        else:
            exit_signal = True

    exit_signal = True
    zed.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/usr/local/zed/volleyball-int8-416.engine', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--ip', type=str, default=None, help='IP address for remote streaming (e.g., 192.168.8.14:30000)')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
