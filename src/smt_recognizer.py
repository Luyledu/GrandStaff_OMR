"""
Sheet Music Transformer识别模块
用于识别谱表图像中的音乐符号
"""

import torch
import cv2
import os
import sys
from pathlib import Path

# 尝试导入SMT相关模块
try:
    from data_augmentation.data_augmentation import convert_img_to_tensor
    from smt_model import SMTModelForCausalLM

    SMT_AVAILABLE = True
except ImportError:
    print("⚠️  SMT相关模块不可用，识别功能将受限")
    SMT_AVAILABLE = False


class SMTRecognizer:
    """Sheet Music Transformer识别器"""

    def __init__(self, model_path, device='auto', max_length=512, beam_size=5):
        """
        初始化SMT识别器

        Args:
            model_path: 模型路径
            device: 运行设备
            max_length: 最大序列长度
            beam_size: beam search大小
        """
        if not SMT_AVAILABLE:
            raise ImportError("SMT相关模块不可用，请确保data_augmentation和smt_model可用")

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.max_length = max_length
        self.beam_size = beam_size

        try:
            # 加载模型
            self.model = SMTModelForCausalLM.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
            print(f"   SMT模型加载成功 (设备: {device})")
        except Exception as e:
            raise RuntimeError(f"SMT模型加载失败: {e}")

    def recognize(self, image_path):
        """
        识别谱表图像中的音乐符号

        Args:
            image_path: 谱表图像路径

        Returns:
            str: 识别的音乐符号编码
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        try:
            # 转换为模型输入张量
            input_tensor = convert_img_to_tensor(image).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                predictions, _ = self.model.predict(
                    input_tensor,
                    convert_to_str=True,
                    max_length=self.max_length,
                    beam_size=self.beam_size
                )

            # 处理预测结果
            if predictions and len(predictions) > 0:
                result = "".join(predictions)
                # 替换特殊标记
                result = result.replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
                return result.strip()
            else:
                return ""

        except Exception as e:
            raise RuntimeError(f"SMT识别失败: {e}")

    def recognize_batch(self, image_paths):
        """
        批量识别谱表图像

        Args:
            image_paths: 谱表图像路径列表

        Returns:
            list: 识别结果列表
        """
        results = []

        for i, image_path in enumerate(image_paths):
            try:
                result = self.recognize(image_path)
                results.append({
                    'image_path': image_path,
                    'result': result,
                    'success': True
                })
                print(f"   批量处理 {i + 1}/{len(image_paths)}: 成功")
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'result': f"识别失败: {str(e)}",
                    'success': False
                })
                print(f"   批量处理 {i + 1}/{len(image_paths)}: 失败 - {e}")

        return results