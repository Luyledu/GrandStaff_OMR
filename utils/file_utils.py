"""
文件工具模块
提供文件操作相关的辅助函数
"""

import os
import shutil
from pathlib import Path


def ensure_dir(path):
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_temp_files(temp_dir):
    """
    清理临时文件

    Args:
        temp_dir: 临时目录路径
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"临时文件已清理: {temp_dir}")
        except Exception as e:
            print(f"清理临时文件失败: {e}")


def get_image_files(input_path, extensions=None):
    """
    获取指定目录中的所有图像文件

    Args:
        input_path: 输入路径（文件或目录）
        extensions: 支持的文件扩展名

    Returns:
        list: 图像文件路径列表
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    image_files = []

    if os.path.isfile(input_path):
        # 单文件
        ext = os.path.splitext(input_path)[1].lower()
        if ext in extensions:
            image_files.append(input_path)
    elif os.path.isdir(input_path):
        # 目录
        for root, dirs, files in os.walk(input_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    image_files.append(os.path.join(root, file))

    # 按文件名排序
    image_files.sort()

    return image_files


def save_json(data, file_path, indent=2):
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    ensure_dir(os.path.dirname(file_path))

    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path):
    """
    从JSON文件加载数据

    Args:
        file_path: 文件路径

    Returns:
        dict: 加载的数据
    """
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)