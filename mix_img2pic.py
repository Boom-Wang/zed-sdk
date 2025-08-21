import os
import shutil
from pathlib import Path

def get_next_number(output_dir):
    """
    获取输出文件夹中下一个可用的编号
    扫描output_dir中所有v_开头的文件，找到最大的编号
    """
    max_num = 0
    
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 1
    
    # 扫描输出文件夹中的所有文件
    for filename in os.listdir(output_dir):
        # 检查是否是left开头的文件
        if filename.startswith('left') and '.' in filename:
            try:
                # 提取文件名中的数字部分
                name_without_ext = filename.split('.')[0]
                num_str = name_without_ext.split('left')[1]
                num = int(num_str)
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                # 如果解析失败，跳过这个文件
                continue
    
    return max_num + 1

def move_images_from_folder(input_img_path, output_img_path):
    """
    将input_img_path文件夹中的所有图片剪切到output_img_path
    并按照v_1, v_2...的格式重命名
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_img_path):
        print(f"错误：输入文件夹 '{input_img_path}' 不存在")
        return 0
    
    # 获取起始编号
    start_num = get_next_number(output_img_path)
    
    # 获取输入文件夹中的所有图片文件
    image_files = []
    for filename in os.listdir(input_img_path):
        file_path = os.path.join(input_img_path, filename)
        if os.path.isfile(file_path):
            # 检查文件扩展名是否为图片格式
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append((filename, ext))
    
    # 如果没有找到图片文件
    if not image_files:
        print(f"警告：在文件夹 '{input_img_path}' 中没有找到图片文件")
        return 0
    
    # 剪切并重命名文件
    moved_count = 0
    current_num = start_num
    
    for filename, ext in image_files:
        old_path = os.path.join(input_img_path, filename)
        new_filename = f"left_{current_num}{ext}"
        new_path = os.path.join(output_img_path, new_filename)
        
        try:
            shutil.move(old_path, new_path)
            print(f"已剪切: {filename} -> {new_filename}")
            moved_count += 1
            current_num += 1
        except Exception as e:
            print(f"错误：无法剪切文件 {filename}. 原因: {str(e)}")
    
    return moved_count

# 便捷函数：直接处理单个文件夹
def process_single_folder():
    """
    便捷函数：处理单个文件夹的简化版本
    """
    # 设置路径
    input_img_path = '/home/laplace/data/ZED-img-Volleyball-SVGA2'
    output_img_path = '/home/laplace/data/ZED-img-Volleyball-SVGA'
    
    # 执行剪切
    count = move_images_from_folder(input_img_path, output_img_path)
    print(f"\n操作完成！成功剪切了 {count} 个图片文件")

# 便捷函数：批量处理多个文件夹
def process_multiple_folders():
    """
    便捷函数：交互式处理多个文件夹
    """
    output_img_path = "/home/laplace/data/V-ball-8-20"
    
    input_folders = [
        "/home/laplace/data/ZED-SVGA-Volleyball",
        "/home/laplace/data/ZED-SVGA-Volleyball1"
    ]

    # 处理所有文件夹
    total_moved = 0
    for idx, input_img_path in enumerate(input_folders, 1):
        print(f"\n处理第 {idx}/{len(input_folders)} 个文件夹: {input_img_path}")
        count = move_images_from_folder(input_img_path, output_img_path)
        total_moved += count
    
    print(f"\n操作完成！总共剪切了 {total_moved} 个图片文件")

if __name__ == "__main__":
    # 运行主函数示例
    # main()
    
    # 或者使用交互式模式
    print("图片批量剪切与重命名工具")
    print("1. 处理单个文件夹")
    print("2. 处理多个文件夹")
    choice = input("\n请选择操作模式 (1 或 2): ").strip()
    
    if choice == "1":
        process_single_folder()
    elif choice == "2":
        process_multiple_folders()
    else:
        print("无效的选择")