########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 
import os

# ... [on_mouse 和其他全局变量保持不变] ...
camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1,-1 )

def on_mouse(event,x,y,flags,param):
    global select_in_progress,selection_rect,origin_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        origin_rect = (x, y)
        select_in_progress = True
    elif event == cv2.EVENT_LBUTTONUP:
        select_in_progress = False 
    elif event == cv2.EVENT_RBUTTONDOWN:
        select_in_progress = False 
        selection_rect = sl.Rect(0,0,0,0)
    
    if select_in_progress:
        selection_rect.x = min(x,origin_rect[0])
        selection_rect.y = min(y,origin_rect[1])
        selection_rect.width = abs(x-origin_rect[0])+1
        selection_rect.height = abs(y-origin_rect[1])+1

def main(opt):

    base_save_path = "/home/laplace/data/zed-get-imgs"
    save_path = base_save_path
    counter = 1

    # 循环检查文件夹是否存在，如果存在则在末尾添加序号
    while os.path.exists(save_path):
        save_path = f"{base_save_path}_{counter}"
        counter += 1

    # 创建找到的唯一文件夹
    os.makedirs(save_path)
    print(f"图片将保存在新建的文件夹中: '{save_path}'")

    image_counter = 1

    STATE_WAITING = 0
    STATE_CAPTURING = 1
    current_state = STATE_WAITING

    FRAMES_PER_BURST = 30
    burst_frame_count = 0
    
    CAPTURE_INTERVAL_MS = 0
    last_burst_start_time = 0

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_fps = 60
    init.sdk_verbose = 1

    if opt.ip_address:
        print(f"正在以IP推流模式启动，连接到: {opt.ip_address}")
        init.set_from_stream(opt.ip_address.split(':')[0], int(opt.ip_address.split(':')[1]))
    else:
        print("未提供IP地址，正在以本地相机模式启动...")
        init.camera_resolution = sl.RESOLUTION.HD1080

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("相机打开失败: "+repr(status)+". 程序退出。")
        exit()
        
    runtime = sl.RuntimeParameters()
    win_name = "Camera Control | Press 'q' to exit"
    mat = sl.Mat()
    
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name,on_mouse)
    
    print_camera_information(cam)
    print_help()
    switch_camera_settings()
    
    key = ''
    
    while key != 113:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.SIDE_BY_SIDE)
            cvImage = mat.get_data()
            timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            if current_state == STATE_WAITING:
                if timestamp - last_burst_start_time >= CAPTURE_INTERVAL_MS:
                    current_state = STATE_CAPTURING
                    burst_frame_count = 0
                    last_burst_start_time = timestamp
                    print(f"\n开始新的采集脉冲 (目标 {FRAMES_PER_BURST} 帧)...")

            if current_state == STATE_CAPTURING:
                if burst_frame_count < FRAMES_PER_BURST:
                    image_name = f"{opt.prefix}_{image_counter}.png"
                    full_path = os.path.join(save_path, image_name)
                    cv2.imwrite(full_path, cvImage)
                    
                    image_counter += 1
                    burst_frame_count += 1
                    print(f"  已采集脉冲帧: {burst_frame_count}/{FRAMES_PER_BURST}")
                else:
                    print("采集脉冲完成。返回等待状态...")
                    current_state = STATE_WAITING
            if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0,0,cvImage.shape[1],cvImage.shape[0]))):
                cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y),(selection_rect.width+selection_rect.x,selection_rect.height+selection_rect.y),(220, 180, 20), 2)
            
            cv2.imshow(win_name, cvImage)
        else:
            print("捕获过程中出错: ", err)
            break
            
        key = cv2.waitKey(1)
        update_camera_settings(key, cam, runtime, mat)
        
    cv2.destroyAllWindows()
    cam.close()
    print(f"\n相机已关闭。总共保存了 {image_counter - 1} 张图片。")

# ... [print_camera_information, print_help, update_camera_settings, switch_camera_settings, valid_ip_or_hostname 函数保持不变] ...
def print_camera_information(cam):
    cam_info = cam.get_camera_information()
    print("ZED Model                 : {0}".format(cam_info.camera_model))
    print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version,cam_info.sensors_configuration.firmware_version))
    print("ZED Camera Resolution     : {0}x{1}".format(round(cam_info.camera_configuration.resolution.width, 2), cam.get_camera_information().camera_configuration.resolution.height))
    print("ZED Camera FPS            : {0}".format(int(cam_info.camera_configuration.fps)))

def print_help():
    print("\n\n相机控制热键:")
    print("* 增加相机设置值:  '+'")
    print("* 减少相机设置值:  '-'")
    print("* 切换相机设置项:  's'")
    print("* 切换相机LED灯:   'l' (小写L)")
    print("* 重置所有参数:    'r'")
    print("* 重置曝光区域:    'f'")
    print("* 使用鼠标选择曝光区域后按 'a' 应用")
    print("* 退出:            'q'\n")

def update_camera_settings(key, cam, runtime, mat):
    global led_on
    if key == 115: switch_camera_settings()
    elif key == 43:
        current_value = cam.get_camera_settings(camera_settings)[1]
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:
        current_value = cam.get_camera_settings(camera_settings)[1]
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:
        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("[示例] 所有设置已重置为默认值")
    elif key == 108:
        led_on = not led_on
        cam.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, led_on)
    elif key == 97 :
        print("[示例] 在目标区域设置AEC_AGC_ROI [",selection_rect.x,",",selection_rect.y,",",selection_rect.width,",",selection_rect.height,"]")
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI,selection_rect,sl.SIDE.BOTH)
    elif key == 102:
        print("[示例] 重置AEC_AGC_ROI为全分辨率")
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI,selection_rect,sl.SIDE.BOTH,True)

def switch_camera_settings():
    global camera_settings, str_camera_settings
    settings = {
        sl.VIDEO_SETTINGS.BRIGHTNESS: (sl.VIDEO_SETTINGS.CONTRAST, "Contrast"),
        sl.VIDEO_SETTINGS.CONTRAST: (sl.VIDEO_SETTINGS.HUE, "Hue"),
        sl.VIDEO_SETTINGS.HUE: (sl.VIDEO_SETTINGS.SATURATION, "Saturation"),
        sl.VIDEO_SETTINGS.SATURATION: (sl.VIDEO_SETTINGS.SHARPNESS, "Sharpness"),
        sl.VIDEO_SETTINGS.SHARPNESS: (sl.VIDEO_SETTINGS.GAIN, "Gain"),
        sl.VIDEO_SETTINGS.GAIN: (sl.VIDEO_SETTINGS.EXPOSURE, "Exposure"),
        sl.VIDEO_SETTINGS.EXPOSURE: (sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, "White Balance"),
        sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE: (sl.VIDEO_SETTINGS.BRIGHTNESS, "Brightness")
    }
    camera_settings, str_camera_settings = settings[camera_settings]
    print(f"[示例] 切换到相机设置: {str_camera_settings.upper()}")

def valid_ip_or_hostname(ip_or_hostname):
    try:
        host, port = ip_or_hostname.split(':')
        socket.inet_aton(host)
        port = int(port)
        return f"{host}:{port}"
    except (socket.error, ValueError):
        raise argparse.ArgumentTypeError("无效的IP地址或主机名格式。请使用 a.b.c.d:p 或 hostname:p 格式")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_address', type=valid_ip_or_hostname, help='(可选) 发送端的IP地址和端口。格式: a.b.c.d:p。如果留空，则使用本地相机。')
    parser.add_argument('--prefix', type=str, default='image', help='(可选) 保存图片的文件名前缀。默认为 "image"。')
    opt = parser.parse_args()
    main(opt)