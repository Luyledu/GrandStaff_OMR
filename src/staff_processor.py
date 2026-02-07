"""
谱表后处理模块
用于处理YOLO检测到的大谱表区域
"""

import cv2
import numpy as np
from pathlib import Path


class StaffProcessor:
    """谱表后处理器"""

    def __init__(self, target_label="B_u", target_height=256,
                 conf_threshold=0.5, margin=30):
        """
        初始化谱表处理器

        Args:
            target_label: 目标标签名称
            target_height: 目标高度
            conf_threshold: 置信度阈值
            margin: 边缘余量
        """
        self.target_label = target_label
        self.target_height = target_height
        self.conf_threshold = conf_threshold
        self.margin = margin

    def process_regions(self, image_path, detections):
        """
        处理所有检测到的谱表区域

        Args:
            image_path: 原始图像路径
            detections: YOLO检测结果列表

        Returns:
            list: 处理后的谱表区域
        """
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 筛选目标标签的区域
        target_regions = [
            det for det in detections
            if det['label'] == self.target_label and det['confidence'] >= self.conf_threshold
        ]

        if not target_regions:
            return []

        # 按垂直位置排序（从上到下）
        target_regions.sort(key=lambda r: r['bbox'][1])

        # 处理每个区域
        processed_regions = []
        for i, region in enumerate(target_regions):
            try:
                processed_region = self._process_single_region(original_image, region, i)
                if processed_region:
                    processed_regions.append(processed_region)
            except Exception as e:
                print(f"⚠️  处理区域 {i} 失败: {e}")
                continue

        return processed_regions

    def _process_single_region(self, original_image, region, region_id):
        """
        处理单个谱表区域

        Args:
            original_image: 原始图像
            region: 区域信息
            region_id: 区域ID

        Returns:
            dict: 处理后的区域信息
        """
        # 获取边界框
        x1, y1, x2, y2 = region['bbox']
        img_h, img_w = original_image.shape[:2]

        # 添加边缘余量
        x1 = max(0, x1 - self.margin)
        y1 = max(0, y1 - self.margin)
        x2 = min(img_w, x2 + self.margin)
        y2 = min(img_h, y2 + self.margin)

        # 确保区域有效
        if x2 <= x1 or y2 <= y1:
            return None

        # 裁剪区域
        region_image = original_image[y1:y2, x1:x2]

        # 检查图像是否有效
        if region_image.size == 0:
            return None

        return {
            'id': region_id,
            'bbox': [x1, y1, x2, y2],
            'confidence': region['confidence'],
            'image': region_image,
            'original_bbox': region['bbox']
        }

    def correct_skew(self, image):
        """
        校正图像倾斜

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 校正后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 二值化
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None:
            return image

        # 计算平均角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -10 < angle < 10:  # 只考虑接近水平的线
                angles.append(angle)

        if not angles:
            return image

        avg_angle = np.mean(angles)

        # 如果角度太小，不进行旋转
        if abs(avg_angle) < 0.5:
            return image

        # 旋转图像
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated