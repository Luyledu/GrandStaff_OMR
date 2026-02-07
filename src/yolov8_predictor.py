"""
YOLO11分割模块
用于检测大谱表区域
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path


class YOLOv8Predictor:
    """YOLO分割器封装类"""

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, device='auto'):
        """
        初始化YOLO分割器

        Args:
            model_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 运行设备 ('auto', 'cpu', 'cuda')
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            print(f"   YOLO模型加载成功 (设备: {device})")
        except Exception as e:
            raise RuntimeError(f"YOLO模型加载失败: {e}")

    def predict(self, image_path):
        """
        对输入图像进行大谱表检测

        Args:
            image_path: 输入图像路径

        Returns:
            dict: 检测结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        original_h, original_w = image.shape[:2]

        try:
            # 使用YOLO进行推理
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    # 获取类别名称
                    class_names = result.names

                    for box in result.boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = class_names[cls_id]

                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, original_w - 1))
                        y1 = max(0, min(y1, original_h - 1))
                        x2 = max(0, min(x2, original_w - 1))
                        y2 = max(0, min(y2, original_h - 1))

                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'label': label,
                            'class_id': cls_id,
                            'area': (x2 - x1) * (y2 - y1)
                        })

            return {
                'image_path': image_path,
                'image_size': (original_w, original_h),
                'detections': detections,
                'detection_count': len(detections)
            }

        except Exception as e:
            raise RuntimeError(f"YOLO推理失败: {e}")

    def visualize_detections(self, image_path, output_path=None):
        """
        可视化检测结果

        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径

        Returns:
            np.ndarray: 可视化图像
        """
        result = self.predict(image_path)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        # 绘制检测框
        for detection in result['detections']:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            label = detection['label']

            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label_text = f"{label}: {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # 标签背景
            cv2.rectangle(image, (x1, y1 - text_height - 4),
                          (x1 + text_width, y1), (0, 255, 0), -1)

            # 标签文字
            cv2.putText(image, label_text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image