#!/usr/bin/env python3
"""
测试端到端OMR系统
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")

    try:
        from src.yolov8_predictor import YOLOv8Predictor
        print("✅ YOLOv8Predictor 导入成功")
    except ImportError as e:
        print(f"❌ YOLOv8Predictor 导入失败: {e}")

    try:
        from src.staff_processor import StaffProcessor
        print("✅ StaffProcessor 导入成功")
    except ImportError as e:
        print(f"❌ StaffProcessor 导入失败: {e}")

    try:
        from src.image_resizer import ImageResizer
        print("✅ ImageResizer 导入成功")
    except ImportError as e:
        print(f"❌ ImageResizer 导入失败: {e}")

    try:
        from src.result_integrator import ResultIntegrator
        print("✅ RESULTIntegrator 导入成功")
    except ImportError as e:
        print(f"❌ RESULTIntegrator 导入失败: {e}")

    try:
        from utils.file_utils import ensure_dir
        print("✅ file_utils 导入成功")
    except ImportError as e:
        print(f"❌ file_utils 导入失败: {e}")

    print("\n模块导入测试完成!")


def test_config():
    """测试配置文件"""
    print("\n测试配置文件...")

    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"✅ 配置文件存在: {config_path}")

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ['yolo', 'postprocess', 'smt', 'output', 'paths']
        for key in required_keys:
            if key in config:
                print(f"  ✅ {key} 配置存在")
            else:
                print(f"  ❌ {key} 配置缺失")
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        print("  请运行: python main.py --input test.jpg 自动创建配置")


def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")

    required_dirs = ['src', 'utils', 'models', 'results']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ 目录存在: {dir_name}")
        else:
            print(f"⚠️  目录不存在: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
            print(f"  已创建: {dir_name}")


def test_dependencies():
    """测试依赖包"""
    print("\n测试依赖包...")

    dependencies = [
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('opencv', 'cv2'),
        ('numpy', 'numpy'),
        ('PIL', 'PIL'),
        ('yaml', 'yaml'),
    ]

    for display_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {display_name} 可用")
        except ImportError:
            print(f"❌ {display_name} 不可用")


if __name__ == "__main__":
    print("=" * 60)
    print("端到端OMR系统测试")
    print("=" * 60)

    test_imports()
    test_config()
    test_directory_structure()
    test_dependencies()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 将模型文件放入 models/ 目录")
    print("2. 运行: python main.py --input test.jpg")
    print("3. 查看 results/ 目录中的输出")