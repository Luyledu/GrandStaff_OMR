"""
结果整合模块
用于整合所有区域的识别结果
"""

import os
import json
from typing import List, Dict, Any


class ResultIntegrator:
    """结果整合器"""

    def __init__(self):
        pass

    def integrate(self, region_results):
        """
        整合所有区域的识别结果

        Args:
            region_results: 区域识别结果列表

        Returns:
            str: 整合后的结果
        """
        if not region_results:
            return ""

        # 按区域ID排序
        sorted_results = sorted(region_results, key=lambda x: x.get('region_id', 0))

        # 构建整合结果
        integrated_lines = []

        for i, result in enumerate(sorted_results):
            region_id = result.get('region_id', i)
            krn_text = result.get('krn_text', '')

            # 添加区域分隔标记
            if i > 0:
                integrated_lines.append(f"\n{'=' * 40}")
                integrated_lines.append(f"Region {region_id}")
                integrated_lines.append(f"{'=' * 40}\n")
            else:
                integrated_lines.append(f"{'=' * 40}")
                integrated_lines.append(f"Region {region_id}")
                integrated_lines.append(f"{'=' * 40}\n")

            # 添加识别结果
            if krn_text:
                integrated_lines.append(krn_text)
            else:
                integrated_lines.append("(无识别结果)")

        return "\n".join(integrated_lines)

    def integrate_from_files(self, krn_files, output_file=None):
        """
        从KRN文件整合结果

        Args:
            krn_files: KRN文件路径列表
            output_file: 输出文件路径

        Returns:
            str: 整合后的结果
        """
        all_results = []

        for i, krn_file in enumerate(krn_files):
            try:
                with open(krn_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                if content:
                    all_results.append({
                        'region_id': i,
                        'krn_text': content,
                        'file_path': krn_file
                    })
            except Exception as e:
                print(f"⚠️  读取文件失败 {krn_file}: {e}")
                all_results.append({
                    'region_id': i,
                    'krn_text': f"文件读取失败: {e}",
                    'file_path': krn_file
                })

        # 整合结果
        integrated_result = self.integrate(all_results)

        # 保存到文件
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(integrated_result)
            print(f"整合结果已保存: {output_file}")

        return integrated_result

    def export_to_musicxml(self, krn_text, output_path):
        """
        导出为MusicXML格式（需要music21支持）

        Args:
            krn_text: KRN文本
            output_path: 输出文件路径
        """
        try:
            import music21

            # 这里需要根据实际的KRN格式进行解析
            # 由于KRN格式复杂，这里只是一个示例
            print("⚠️  MusicXML导出功能需要根据KRN格式具体实现")
            print(f"   原始KRN内容长度: {len(krn_text)} 字符")

            # 保存原始KRN文件
            krn_path = output_path.replace('.xml', '.krn')
            with open(krn_path, 'w', encoding='utf-8') as f:
                f.write(krn_text)

            return krn_path

        except ImportError:
            print("⚠️  music21模块未安装，无法导出MusicXML")
            print("   请安装: pip install music21")
            return None