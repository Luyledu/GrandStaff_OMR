"""
可视化模块
用于生成处理过程的可视化结果
"""

import cv2
import numpy as np


def visualize_detections(image, detections, output_path=None):
    """
    可视化检测结果

    Args:
        image: 原始图像
        detections: 检测结果列表
        output_path: 输出路径

    Returns:
        np.ndarray: 可视化图像
    """
    vis_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['label']
        conf = detection['confidence']

        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签
        label_text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        # 标签背景
        cv2.rectangle(vis_image, (x1, y1 - text_height - 4),
                      (x1 + text_width, y1), (0, 255, 0), -1)

        # 标签文字
        cv2.putText(vis_image, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, vis_image)

    return vis_image