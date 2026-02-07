import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from typing import List, Dict, Tuple


class StaffLineDetector:
    """五线谱检测器，负责检测和校正五线谱"""

    def __init__(self):
        self.staff_spacing = 0
        self.staff_thickness = 0

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像，增强对比度并二值化"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 高斯模糊减少噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用自适应阈值进行二值化
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        return thresh

    def detect_staff_lines(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """检测五线谱位置和参数"""
        thresh = self.preprocess_image(image)

        # 垂直投影分析找到五线谱区域
        vertical_projection = np.sum(thresh, axis=1) / 255
        avg_projection = np.mean(vertical_projection)

        # 找到可能的五线谱区域
        staff_regions = []
        in_staff = False
        staff_start = 0

        for i, proj in enumerate(vertical_projection):
            if proj > avg_projection * 0.7 and not in_staff:
                in_staff = True
                staff_start = i
            elif proj < avg_projection * 0.3 and in_staff:
                in_staff = False
                if i - staff_start > 50:  # 最小高度阈值
                    staff_regions.append((staff_start, i))

        # 分析五线谱参数
        if staff_regions:
            sample_region = staff_regions[0]
            region_height = sample_region[1] - sample_region[0]
            self.staff_spacing = region_height / 9  # 五线四间
            self.staff_thickness = int(self.staff_spacing * 0.15)

        return staff_regions

    def remove_staff_lines(self, image: np.ndarray, staff_regions: List[Tuple[int, int]]) -> np.ndarray:
        """从图像中移除五线谱"""
        thresh = self.preprocess_image(image)
        result = thresh.copy()

        for region in staff_regions:
            y_start, y_end = region

            # 检测每条线
            for i in range(5):
                line_y = int(y_start + self.staff_spacing * (2 * i + 1) / 2)

                # 水平投影分析找到线的长度
                line_projection = np.sum(thresh[line_y - self.staff_thickness:line_y + self.staff_thickness, :],
                                         axis=0) / 255
                line_indices = np.where(line_projection > self.staff_thickness * 0.8)[0]

                if len(line_indices) > 0:
                    x_start = min(line_indices)
                    x_end = max(line_indices)

                    # 移除检测到的线
                    cv2.rectangle(result, (x_start, line_y - self.staff_thickness),
                                  (x_end, line_y + self.staff_thickness), 0, -1)

        return result


class ClefDetector:
    """连谱号检测器，使用YOLOv8模型"""

    def __init__(self, model_path: str = 'clef_detection_model.pt'):
        """初始化连谱号检测器"""
        self.model = YOLO(model_path)
        self.clef_types = ['treble', 'bass', 'alto', 'tenor']

    def detect_clef(self, image: np.ndarray, staff_regions: List[Tuple[int, int]]) -> List[Dict]:
        """检测五线谱区域中的连谱号"""
        clef_positions = []

        for region in staff_regions:
            y_start, y_end = region
            staff_height = y_end - y_start

            # 提取可能包含连谱号的区域（左侧部分）
            region_width = min(int(staff_height * 3), image.shape[1])
            clef_region = image[y_start:y_end, 0:region_width]

            # 使用YOLOv8进行检测
            results = self.model(clef_region, classes=[0, 1, 2, 3], conf=0.5)  # 假设连谱号类别ID为0-3

            # 处理检测结果
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls)
                    confidence = box.conf

                    if confidence > 0.5:  # 置信度阈值
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        # 转换为原图坐标
                        x1_original = x1
                        y1_original = y1 + y_start
                        x2_original = x2
                        y2_original = y2 + y_start

                        clef_positions.append({
                            'type': self.clef_types[class_id],
                            'bbox': (x1_original, y1_original, x2_original, y2_original),
                            'confidence': float(confidence),
                            'staff_region': region
                        })

        return clef_positions


class MeasureLineDetector:
    """小节线检测器，用于识别乐谱中的小节线"""

    def __init__(self):
        pass

    def detect_measure_lines(self, image: np.ndarray, staff_regions: List[Tuple[int, int]]) -> Dict[
        Tuple[int, int], List[int]]:
        """检测每个五线谱区域中的小节线"""
        measure_lines = {}

        for region in staff_regions:
            y_start, y_end = region
            staff_height = y_end - y_start

            # 提取五线谱区域
            staff_area = image[y_start:y_end, :]

            # 垂直投影分析找到小节线
            vertical_projection = np.sum(staff_area, axis=0) / 255
            threshold = np.max(vertical_projection) * 0.7

            # 寻找高于阈值的连续区域（小节线）
            lines = []
            in_line = False
            line_start = 0

            for i, proj in enumerate(vertical_projection):
                if proj > threshold and not in_line:
                    in_line = True
                    line_start = i
                elif proj < threshold and in_line:
                    in_line = False
                    line_width = i - line_start

                    # 过滤掉太窄或太宽的区域
                    if staff_height * 0.5 < line_width < staff_height * 3:
                        line_center = line_start + line_width // 2
                        lines.append(line_center)

            measure_lines[region] = lines

        return measure_lines


class StaffSegmenter:
    """五线谱分割器，根据连谱号和小节线将五线谱分割成逻辑单元"""

    def __init__(self, clef_model_path: str = 'clef_detection_model.pt'):
        self.staff_detector = StaffLineDetector()
        self.clef_detector = ClefDetector(clef_model_path)
        self.measure_detector = MeasureLineDetector()

    def segment(self, image_path: str) -> List[Dict]:
        """分割五线谱图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            # 尝试获取更多错误信息
            try:
                with open(image_path, 'rb') as f:
                    # 检查文件头部，确认是否为图像
                    header = f.read(10)
                    print(f"文件头部信息: {header}")
                    if b'\xFF\xD8' in header:
                        print("文件似乎是JPEG格式")
                    elif b'\x89PNG' in header:
                        print("文件似乎是PNG格式")
                    else:
                        print("未知文件格式")
            except Exception as e:
                print(f"无法打开文件: {e}")
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 1. 检测五线谱
        staff_regions = self.staff_detector.detect_staff_lines(image)
        if not staff_regions:
            print(f"警告: 图像 {image_path} 中未检测到五线谱区域")
            return []

        # 2. 检测连谱号
        clef_positions = self.clef_detector.detect_clef(image, staff_regions)

        # 3. 检测小节线
        measure_lines = self.measure_detector.detect_measure_lines(
            self.staff_detector.remove_staff_lines(image, staff_regions),
            staff_regions
        )

        # 4. 按连谱号和小节线分割
        segments = self._create_segments(image, staff_regions, clef_positions, measure_lines)

        return segments

    def _create_segments(self, image: np.ndarray, staff_regions: List[Tuple[int, int]],
                         clef_positions: List[Dict], measure_lines: Dict[Tuple[int, int], List[int]]) -> List[Dict]:
        """根据连谱号和小节线创建分割片段"""
        segments = []

        for region in staff_regions:
            y_start, y_end = region

            # 获取当前五线谱区域的连谱号
            region_clefs = [c for c in clef_positions if c['staff_region'] == region]
            # 按x坐标排序
            region_clefs.sort(key=lambda c: c['bbox'][0])

            # 获取当前五线谱区域的小节线
            region_measures = measure_lines.get(region, [])

            # 如果没有检测到连谱号，将整个区域作为一个片段
            if not region_clefs:
                segments.append({
                    'type': 'full_staff',
                    'image': image[y_start:y_end, :],
                    'position': (0, y_start),
                    'size': (image.shape[1], y_end - y_start),
                    'staff_region': region
                })
                continue

            # 处理每个连谱号及其后续区域
            for i, clef in enumerate(region_clefs):
                clef_x1, _, clef_x2, _ = clef['bbox']

                # 确定当前片段的结束位置
                if i < len(region_clefs) - 1:
                    # 如果不是最后一个连谱号，下一个连谱号之前的区域
                    next_clef_x1 = region_clefs[i + 1]['bbox'][0]
                    end_x = next_clef_x1
                else:
                    # 最后一个连谱号，到图像末尾或最后一个小节线
                    if region_measures:
                        end_x = max(region_measures) + 20  # 加上一点边距
                    else:
                        end_x = image.shape[1]

                # 确定当前片段的开始位置（连谱号右侧）
                start_x = clef_x2

                # 找到开始位置和结束位置之间的小节线
                relevant_measures = [m for m in region_measures if start_x < m < end_x]
                # 添加开始和结束位置
                measure_points = [start_x] + relevant_measures + [end_x]

                # 根据小节线进一步分割
                for j in range(len(measure_points) - 1):
                    seg_start = measure_points[j]
                    seg_end = measure_points[j + 1]

                    # 创建片段
                    segment_type = 'measure' if j > 0 else 'clef_and_first_measure'

                    segments.append({
                        'type': segment_type,
                        'image': image[y_start:y_end, seg_start:seg_end],
                        'position': (seg_start, y_start),
                        'size': (seg_end - seg_start, y_end - y_start),
                        'staff_region': region,
                        'clef_info': clef if j == 0 else None
                    })

        return segments


def visualize_segments(segments: List[Dict], output_dir: str):
    """可视化并保存分割结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 8))

    for i, segment in enumerate(segments):
        plt.subplot(len(segments), 1, i + 1)
        plt.title(f"Segment {i + 1} ({segment['type']})")

        if len(segment['image'].shape) == 3:
            plt.imshow(cv2.cvtColor(segment['image'], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(segment['image'], cmap='gray')

        plt.axis('off')

        # 保存片段
        output_path = os.path.join(output_dir, f"segment_{i}_{segment['type']}.png")
        cv2.imwrite(output_path, segment['image'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_result.png'))
    plt.close()


def is_image_file(file_path: str) -> bool:
    """检查文件是否为图像文件"""
    if not os.path.isfile(file_path):
        return False

    # 检查文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ext = os.path.splitext(file_path)[1].lower()

    if ext in image_extensions:
        # 尝试读取文件头部确认
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if (b'\xFF\xD8' in header) or (b'\x89PNG' in header) or (b'GIF8' in header):
                    return True
        except:
            return False

    return False


def process_single_image(image_path: str, model_path: str, output_base_dir: str):
    """处理单张图像"""
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)

    # 验证输入是否为有效图像文件
    if not is_image_file(image_path):
        print(f"错误: {image_path} 不是有效的图像文件")
        return 0

    print(f"\n=== 处理图像: {image_path} ===")

    # 验证文件是否可读
    try:
        with open(image_path, 'rb') as f:
            pass
    except Exception as e:
        print(f"错误: 无法访问文件 {image_path}: {str(e)}")
        return 0

    # 创建单独的输出目录
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(output_base_dir, image_name)

    # 初始化分割器
    segmenter = StaffSegmenter(clef_model_path=model_path)

    # 分割乐谱
    try:
        segments = segmenter.segment(image_path)

        if segments:
            # 可视化并保存结果
            visualize_segments(segments, output_dir=output_dir)
            print(f"成功分割为 {len(segments)} 个片段，结果保存在: {output_dir}")
            return len(segments)
        else:
            print(f"未能从 {image_path} 分割任何片段")
            return 0
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return 0


def process_directory(input_dir: str, model_path: str, output_base_dir: str):
    """处理目录中的所有图像，包括子目录"""
    # 转换为绝对路径
    input_dir = os.path.abspath(input_dir)

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return 0, 0

    print(f"\n=== 处理目录: {input_dir} ===")

    # 收集所有图像文件（包括子目录）
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path):
                image_files.append(file_path)

    if not image_files:
        print(f"警告: 目录 {input_dir} 及其子目录中未找到图像文件")
        return 0, 0

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    total_images = len(image_files)
    total_segments = 0

    for image_file in image_files:
        segments_count = process_single_image(image_file, model_path, output_base_dir)
        total_segments += segments_count

    print(f"\n处理完成:")
    print(f"总图像数: {total_images}")
    print(f"总分割片段数: {total_segments}")

    return total_images, total_segments


def main():
    """主函数，处理单张图像或目录中的所有图像"""
    # ----------------- 在这里修改文件路径 -----------------
    INPUT_PATH = "/tmp/pycharm_project_588/YOLO11/data/testps/tiankong/images"  # 输入图像路径或目录路径
    CLEF_MODEL_PATH = "/tmp/pycharm_project_588/YOLO11/runs_train/exp22/weights/best.pt"  # 连谱号检测模型路径
    OUTPUT_BASE_DIR = "/tmp/pycharm_project_588/omr.model/SMT-master/Data/yuanimages"  # 输出基础目录
    # ---------------------------------------------------

    # 转换为绝对路径
    input_path = os.path.abspath(INPUT_PATH)
    model_path = os.path.abspath(CLEF_MODEL_PATH)
    output_base_dir = os.path.abspath(OUTPUT_BASE_DIR)

    print(f"\n五线谱分割程序启动")
    print(f"输入路径: {input_path}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_base_dir}")

    # 检查输入路径
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 {input_path} 不存在")
        return

    # 判断是文件还是目录
    if os.path.isfile(input_path):
        # 处理单张图像
        if is_image_file(input_path):
            process_single_image(input_path, model_path, output_base_dir)
        else:
            print(f"错误: {input_path} 不是有效的图像文件")
    elif os.path.isdir(input_path):
        # 处理目录中的所有图像
        process_directory(input_path, model_path, output_base_dir)
    else:
        print(f"错误: 输入路径 {input_path} 既不是文件也不是目录")


if __name__ == "__main__":
    main()