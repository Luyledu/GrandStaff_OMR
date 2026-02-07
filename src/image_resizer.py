"""
图像尺寸调整模块
将图像调整到目标高度
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImageResizer:
    """图像尺寸调整器"""

    def __init__(self, target_height=256):
        """
        初始化图像尺寸调整器

        Args:
            target_height: 目标高度（像素）
        """
        self.target_height = target_height

    def resize(self, image):
        """
        调整图像尺寸

        Args:
            image: 输入图像（numpy数组或PIL图像）

        Returns:
            np.ndarray: 调整后的图像
        """
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # RGB转BGR
                image_np = image_np[:, :, ::-1]
        else:
            image_np = image

        # 获取原始尺寸
        if len(image_np.shape) == 2:
            h, w = image_np.shape
        else:
            h, w = image_np.shape[:2]

        # 检查图像是否有效
        if h == 0 or w == 0:
            raise ValueError("无效的图像尺寸")

        # 计算新宽度，保持宽高比
        scale = self.target_height / h
        new_width = int(w * scale)

        # 调整尺寸
        resized = cv2.resize(
            image_np,
            (new_width, self.target_height),
            interpolation=cv2.INTER_LINEAR
        )

        return resized

    def resize_file(self, input_path, output_path=None):
        """
        调整图像文件尺寸

        Args:
            input_path: 输入图像文件路径
            output_path: 输出图像文件路径

        Returns:
            np.ndarray: 调整后的图像
        """
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            # 尝试用PIL读取
            try:
                pil_image = Image.open(input_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB转BGR
                    image = image[:, :, ::-1]
            except:
                raise FileNotFoundError(f"无法读取图像: {input_path}")

        # 调整尺寸
        resized = self.resize(image)

        # 保存结果
        if output_path:
            self.save_image(resized, output_path)

        return resized

    def save_image(self, image, output_path):
        """
        保存图像到文件

        Args:
            image: 图像数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存图像
        success = cv2.imwrite(output_path, image)
        if not success:
            # 尝试使用PIL保存
            try:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR转RGB
                    image_rgb = image[:, :, ::-1]
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)

                pil_image.save(output_path)
            except Exception as e:
                raise IOError(f"无法保存图像: {output_path} - {e}")