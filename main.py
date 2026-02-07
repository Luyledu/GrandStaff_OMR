#!/usr/bin/env python3
"""
简化版大谱表光学音乐识别系统
只包含：YOLO11分割 → 后处理 → 尺寸调整
移除SMT识别部分，系统更稳定
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
import traceback

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入简化版模块
from src.yolov8_predictor import YOLOv8Predictor
from src.staff_processor import StaffProcessor
from src.image_resizer import ImageResizer
from utils.file_utils import ensure_dir, clean_temp_files
from utils.visualization import visualize_detections


class SimplifiedGrandStaffOMR:
    """简化版大谱表OMR系统（无SMT识别）"""

    def __init__(self, config_path="config.yaml"):
        """初始化系统"""
        self.config = self._load_config(config_path)
        self._init_paths()
        self._init_components()

        print("=" * 60)
        print("🎵 简化版大谱表OMR系统初始化完成")
        print("   功能: YOLO11m分割 + 后处理 + 尺寸调整")
        print(f"   设备: {self.config['yolo']['device']}")
        print(f"   输出目录: {self.config['output']['base_dir']}")
        print("=" * 60)

    def _load_config(self, config_path):
        """加载配置文件"""
        if not os.path.exists(config_path):
            # 创建默认配置
            default_config = self._get_default_config()
            ensure_dir(os.path.dirname(config_path))
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"⚠️ 配置文件不存在，已创建默认配置: {config_path}")
            return default_config

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 设置设备（自动检测CUDA）
        import torch
        device = config['yolo']['device']
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config['yolo']['device'] = device

        return config

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'yolo': {
                'model_path': 'models/yolo11m.pt',
                'input_size': 1280,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'device': 'auto'
            },
            'postprocess': {
                'target_label': 'B_u',
                'target_height': 256,
                'conf_threshold': 0.5,
                'margin': 30,
                'min_expansion': 0.05,
                'max_expansion': 0.3,
                'debug_mode': False
            },
            'output': {
                'base_dir': 'simplified_results',
                'save_intermediate': False,
                'save_visualizations': True,
                'save_cropped_staffs': True
            },
            'paths': {
                'temp_dir': 'temp',
                'intermediate_dir': 'intermediate',
                'final_output_dir': 'cropped_staffs',
                'detection_dir': 'detections'
            }
        }

    def _init_paths(self):
        """初始化路径"""
        base_dir = Path(self.config['output']['base_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.paths = {
            'base': base_dir / timestamp,
            'temp': base_dir / timestamp / self.config['paths']['temp_dir'],
            'intermediate': base_dir / timestamp / self.config['paths']['intermediate_dir'],
            'cropped_staffs': base_dir / timestamp / self.config['paths']['final_output_dir'],
            'detections': base_dir / timestamp / self.config['paths']['detection_dir'],
            'visualizations': base_dir / timestamp / 'visualizations'
        }

        # 创建所有目录
        for path in self.paths.values():
            ensure_dir(path)

    def _init_components(self):
        """初始化所有组件"""
        print("🔄 初始化系统组件...")

        try:
            # 1. YOLO分割器
            print("  1. 加载YOLO11m分割模型...")
            self.yolo_predictor = YOLOv8Predictor(
                model_path=self.config['yolo']['model_path'],
                conf_threshold=self.config['yolo']['conf_threshold'],
                iou_threshold=self.config['yolo']['iou_threshold'],
                device=self.config['yolo']['device']
            )
            print(f"     ✓ YOLO模型加载成功")

            # 2. 谱表处理器
            print("  2. 初始化谱表后处理器...")
            self.staff_processor = StaffProcessor(
                target_label=self.config['postprocess']['target_label'],
                target_height=self.config['postprocess']['target_height'],
                conf_threshold=self.config['postprocess']['conf_threshold'],
                margin=self.config['postprocess']['margin']
            )
            print(f"     ✓ 谱表处理器初始化成功")

            # 3. 图像尺寸调整器
            print("  3. 初始化图像尺寸调整器...")
            self.image_resizer = ImageResizer(
                target_height=self.config['postprocess']['target_height']
            )
            print(f"     ✓ 图像尺寸调整器初始化成功")

            print("✅ 所有组件初始化完成")

        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            traceback.print_exc()
            sys.exit(1)

    def process_single_image(self, image_path):
        """
        处理单张乐谱图像（简化版，无SMT识别）

        Args:
            image_path: 输入图像路径

        Returns:
            dict: 处理结果
        """
        print(f"\n{'=' * 60}")
        print(f"🎵 开始处理: {image_path}")
        print(f"{'=' * 60}")

        start_time = time.time()
        image_name = Path(image_path).stem

        try:
            # 1. YOLO分割
            print(f"\n🔍 步骤1: YOLO大谱表分割...")
            yolo_results = self.yolo_predictor.predict(image_path)

            if not yolo_results['detections']:
                print("⚠️  未检测到大谱表区域")
                return {
                    'success': False,
                    'error': '未检测到大谱表区域',
                    'image_path': image_path
                }

            print(f"    ✓ 检测到 {len(yolo_results['detections'])} 个大谱表区域")

            # 保存检测结果可视化
            if self.config['output']['save_visualizations']:
                print(f"    🎨 生成检测结果可视化...")
                vis_path = self.paths['detections'] / f"{image_name}_detections.png"
                self.yolo_predictor.visualize_detections(image_path, str(vis_path))
                print(f"      ✓ 可视化已保存: {vis_path}")

            # 2. 谱表后处理
            print(f"🎯 步骤2: 谱表区域后处理...")
            staff_regions = self.staff_processor.process_regions(
                image_path,
                yolo_results['detections']
            )

            if not staff_regions:
                print("⚠️  后处理后无有效谱表区域")
                return {
                    'success': False,
                    'error': '后处理后无有效谱表区域',
                    'image_path': image_path
                }

            print(f"    ✓ 后处理完成，生成 {len(staff_regions)} 个谱表区域")

            # 3. 尺寸调整并保存结果
            print(f"📏 步骤3: 尺寸标准化...")
            cropped_results = []

            for i, region in enumerate(staff_regions):
                region_img = region['image']

                # 尺寸调整
                resized_img = self.image_resizer.resize(region_img)

                # 保存调整后的图像
                if self.config['output']['save_cropped_staffs']:
                    output_filename = f"{image_name}_staff_{i:03d}.png"
                    output_path = self.paths['cropped_staffs'] / output_filename
                    self.image_resizer.save_image(resized_img, str(output_path))

                    cropped_results.append({
                        'id': i,
                        'original_bbox': region.get('original_bbox', region['bbox']),
                        'processed_bbox': region['bbox'],
                        'confidence': region['confidence'],
                        'image_path': str(output_path),
                        'image_size': resized_img.shape[:2]
                    })

                    print(f"      ✓ 保存谱表 {i + 1}: {output_filename}")

            # 4. 生成最终可视化结果
            if self.config['output']['save_visualizations']:
                print(f"\n🎨 步骤4: 生成完整可视化结果...")
                self._create_comprehensive_visualization(
                    image_path, yolo_results, staff_regions, image_name
                )

            # 5. 保存处理报告
            print(f"📊 步骤5: 生成处理报告...")
            report_data = self._save_processing_report(
                image_name, yolo_results, staff_regions, cropped_results
            )

            # 计算处理时间
            processing_time = time.time() - start_time

            # 清理临时文件
            if not self.config['output']['save_intermediate']:
                clean_temp_files(str(self.paths['temp']))

            print(f"\n{'=' * 60}")
            print(f"✅ 处理完成!")
            print(f"   处理时间: {processing_time:.2f}秒")
            print(f"   检测区域: {len(yolo_results['detections'])}个")
            print(f"   处理区域: {len(staff_regions)}个")
            print(f"   输出目录: {self.paths['base']}")
            print(f"{'=' * 60}")

            return {
                'success': True,
                'processing_time': processing_time,
                'detections_count': len(yolo_results['detections']),
                'processed_count': len(staff_regions),
                'cropped_count': len(cropped_results),
                'output_dir': str(self.paths['base']),
                'report': report_data
            }

        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }

    def _create_comprehensive_visualization(self, image_path, yolo_results, staff_regions, image_name):
        """创建完整的可视化结果"""
        import cv2
        import numpy as np

        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print("⚠️  无法读取图像进行可视化")
            return

        # 创建可视化图像
        vis_image = image.copy()

        # 绘制YOLO检测框（绿色）
        for detection in yolo_results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            conf = detection['confidence']

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 标签文本
            label_text = f"{label}: {conf:.2f}"
            cv2.putText(vis_image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制后处理区域（红色）
        for i, region in enumerate(staff_regions):
            x1, y1, x2, y2 = region['bbox']

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 区域编号
            cv2.putText(vis_image, f"Staff {i}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 添加标题和说明
        title = f"Grand Staff Detection & Processing: {image_name}"
        cv2.putText(vis_image, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 添加图例
        legend_y = 70
        cv2.putText(vis_image, "Legend:", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, "Green: YOLO Detection", (10, legend_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, "Red: Processed Region", (10, legend_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis_image, f"Total Detections: {len(yolo_results['detections'])}",
                    (10, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Processed Regions: {len(staff_regions)}",
                    (10, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存可视化结果
        vis_path = self.paths['visualizations'] / f"{image_name}_full_visualization.png"
        cv2.imwrite(str(vis_path), vis_image)

        print(f"      ✓ 完整可视化已保存: {vis_path}")

        # 如果检测区域较少，还可以创建并排对比图
        if len(staff_regions) <= 6:
            self._create_side_by_side_visualization(staff_regions, image_name)

    def _create_side_by_side_visualization(self, staff_regions, image_name):
        """创建并排对比可视化"""
        import cv2
        import numpy as np

        # 计算布局
        n_regions = len(staff_regions)
        cols = min(3, n_regions)
        rows = (n_regions + cols - 1) // cols

        # 获取最大尺寸
        max_h, max_w = 0, 0
        for region in staff_regions:
            h, w = region['image'].shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        # 创建画布
        canvas_h = rows * max_h + 50
        canvas_w = cols * max_w + 50
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # 粘贴每个区域
        for i, region in enumerate(staff_regions):
            row = i // cols
            col = i % cols

            img = region['image']
            h, w = img.shape[:2]

            # 计算位置
            x = col * max_w + 25
            y = row * max_h + 25

            # 将图像粘贴到画布上
            canvas[y:y + h, x:x + w] = img

            # 添加标签
            label = f"Staff {i}"
            cv2.putText(canvas, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # 添加边界框信息
            bbox_info = f"Size: {w}x{h}"
            cv2.putText(canvas, bbox_info, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 添加标题
        title = f"Cropped Staff Regions ({n_regions} total)"
        cv2.putText(canvas, title, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 保存
        side_path = self.paths['visualizations'] / f"{image_name}_staff_grid.png"
        cv2.imwrite(str(side_path), canvas)

        print(f"      ✓ 并排对比图已保存: {side_path}")

    def _save_processing_report(self, image_name, yolo_results, staff_regions, cropped_results):
        """保存处理报告"""
        import json

        report_data = {
            'image_name': image_name,
            'processing_time': datetime.now().isoformat(),
            'yolo_detections': {
                'total_count': len(yolo_results['detections']),
                'detections': yolo_results['detections']
            },
            'staff_regions': {
                'total_count': len(staff_regions),
                'regions': [
                    {
                        'id': i,
                        'bbox': region['bbox'],
                        'original_bbox': region.get('original_bbox', region['bbox']),
                        'confidence': region['confidence']
                    }
                    for i, region in enumerate(staff_regions)
                ]
            },
            'cropped_results': cropped_results,
            'output_directories': {
                'base': str(self.paths['base']),
                'cropped_staffs': str(self.paths['cropped_staffs']),
                'visualizations': str(self.paths['visualizations']),
                'detections': str(self.paths['detections'])
            }
        }

        # 保存为JSON
        json_path = self.paths['base'] / f"{image_name}_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 保存为TXT（更易读）
        txt_path = self.paths['base'] / f"{image_name}_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Grand Staff OMR Processing Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Image: {image_name}\n")
            f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"YOLO Detections: {len(yolo_results['detections'])}\n")
            for i, det in enumerate(yolo_results['detections']):
                f.write(f"  {i + 1}. {det['label']}: bbox={det['bbox']}, conf={det['confidence']:.3f}\n")

            f.write(f"\nProcessed Staff Regions: {len(staff_regions)}\n")
            for i, region in enumerate(staff_regions):
                f.write(f"  {i + 1}. bbox={region['bbox']}, conf={region['confidence']:.3f}\n")

            f.write(f"\nCropped Images: {len(cropped_results)}\n")
            for result in cropped_results:
                f.write(f"  Staff {result['id']}: {result['image_path']}\n")

            f.write(f"\nOutput Directories:\n")
            for key, path in report_data['output_directories'].items():
                f.write(f"  {key}: {path}\n")

            f.write(f"\n" + "=" * 60 + "\n")
            f.write("End of Report\n")
            f.write("=" * 60 + "\n")

        print(f"      ✓ 处理报告已保存: {txt_path}")

        return report_data

    def process_batch(self, input_dir):
        """批量处理文件夹中的图像"""
        input_dir = Path(input_dir)

        # 查找所有支持的图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"❌ 未找到支持的图像文件: {input_dir}")
            return []

        print(f"📁 发现 {len(image_files)} 个图像文件")

        # 批量处理
        all_results = []
        for img_path in image_files:
            print(f"\n{'#' * 60}")
            print(f"处理: {img_path.name}")
            print(f"{'#' * 60}")

            result = self.process_single_image(str(img_path))
            all_results.append(result)

            if result['success']:
                print(f"✅ 处理成功")
            else:
                print(f"❌ 处理失败: {result.get('error', '未知错误')}")

        # 生成批量报告
        self._generate_batch_report(all_results)

        return all_results

    def _generate_batch_report(self, results):
        """生成批量处理报告"""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful

        report_path = self.paths['base'] / "batch_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("       批量处理报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文件数: {total}\n")
            f.write(f"成功处理: {successful}\n")
            f.write(f"处理失败: {failed}\n")
            f.write(f"成功率: {successful / total * 100:.1f}%\n\n")

            # 汇总统计
            total_detections = sum(r.get('detections_count', 0) for r in results if r.get('success', False))
            total_processed = sum(r.get('processed_count', 0) for r in results if r.get('success', False))
            total_cropped = sum(r.get('cropped_count', 0) for r in results if r.get('success', False))

            f.write(f"汇总统计:\n")
            f.write(f"  总检测区域数: {total_detections}\n")
            f.write(f"  总处理区域数: {total_processed}\n")
            f.write(f"  总切割图像数: {total_cropped}\n\n")

            if failed > 0:
                f.write("失败文件:\n")
                for result in results:
                    if not result.get('success', False):
                        f.write(f"  - {result.get('image_path', '未知')}: ")
                        f.write(f"{result.get('error', '未知错误')}\n")

        print(f"\n📊 批量处理报告已保存: {report_path}")


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description='简化版大谱表光学音乐识别系统（无SMT识别）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单张图像
  python main.py --input sheet_music.jpg

  # 批量处理文件夹
  python main.py --batch scores_folder/

  # 使用自定义配置
  python main.py --input sheet.jpg --config my_config.yaml
        """
    )

    parser.add_argument('--input', type=str, help='输入图像文件路径')
    parser.add_argument('--batch', type=str, help='批量处理图像文件夹路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型目录路径 (默认: models)')

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        print("\n❌ 错误: 请指定 --input 或 --batch 参数")
        sys.exit(1)

    try:
        # 创建OMR系统实例
        omr_system = SimplifiedGrandStaffOMR(config_path=args.config)

        if args.input:
            # 单张图像处理
            if not os.path.exists(args.input):
                print(f"❌ 错误: 输入文件不存在 - {args.input}")
                sys.exit(1)

            result = omr_system.process_single_image(args.input)

            if result['success']:
                print(f"\n🎉 处理成功!")
                print(f"   输出目录: {omr_system.paths['base']}")
                print(f"   切割图像: {result.get('cropped_count', 0)}个")
            else:
                print(f"\n❌ 处理失败: {result.get('error')}")
                sys.exit(1)

        elif args.batch:
            # 批量处理
            if not os.path.exists(args.batch):
                print(f"❌ 错误: 输入文件夹不存在 - {args.batch}")
                sys.exit(1)

            results = omr_system.process_batch(args.batch)

            successful = sum(1 for r in results if r.get('success', False))
            total = len(results)

            print(f"\n🎉 批量处理完成!")
            print(f"   成功: {successful}/{total} ({successful / total * 100:.1f}%)")
            print(f"   输出目录: {omr_system.paths['base']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()