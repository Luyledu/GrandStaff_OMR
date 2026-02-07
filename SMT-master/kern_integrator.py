import os
from typing import List, Tuple


class KernResultIntegrator:
    """读取模型输出的编码文件，按顺序整合并保存拼接结果"""

    def __init__(self):
        self.krn_contents = []  # 存储读取的编码内容

    def read_krn_file(self, file_path: str) -> bool:
        """读取单个模型输出的Kern编码文件"""
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 - {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # 只保留非空内容
                    self.krn_contents.append(content)
                    return True
                else:
                    print(f"警告: 空文件 - {file_path}")
                    return False
        except Exception as e:
            print(f"读取文件出错 {file_path}: {str(e)}")
            return False

    def integrate_results(self) -> str:
        """按顺序拼接所有读取的编码内容"""
        return ''.join(self.krn_contents)

    def save_integrated_result(self, output_path: str) -> bool:
        """保存整合后的结果到文件"""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 确保文件后缀为.krn
        if not output_path.endswith('.krn'):
            output_path += '.krn'

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.integrate_results())
            return True
        except Exception as e:
            print(f"保存整合结果失败: {str(e)}")
            return False


def get_sorted_krn_files(krn_dir: str) -> List[str]:
    """获取目录中所有Kern编码文件，并按文件名排序"""
    if not os.path.exists(krn_dir):
        raise FileNotFoundError(f"编码文件目录不存在: {krn_dir}")

    # 只筛选.kern和.krn文件
    krn_extensions = ['.krn', '.kern']
    krn_files = [
        os.path.join(krn_dir, f)
        for f in os.listdir(krn_dir)
        if os.path.isfile(os.path.join(krn_dir, f)) and
           os.path.splitext(f)[1].lower() in krn_extensions
    ]

    # 按文件名排序（确保顺序正确）
    krn_files.sort()
    return krn_files


def process_krn_integration(krn_dir: str, output_file: str) -> Tuple[bool, str]:
    """处理整个整合流程：读取编码文件 -> 拼接 -> 保存结果"""
    try:
        # 获取排序后的编码文件列表
        krn_files = get_sorted_krn_files(krn_dir)
        if not krn_files:
            return (False, f"在 {krn_dir} 中未找到任何Kern编码文件")

        print(f"找到 {len(krn_files)} 个编码文件，开始整合...")

        # 初始化整合器并读取所有文件
        integrator = KernResultIntegrator()
        for i, file_path in enumerate(krn_files):
            if integrator.read_krn_file(file_path):
                print(f"已读取 {i + 1}/{len(krn_files)}: {os.path.basename(file_path)}")

        # 保存整合结果
        if integrator.save_integrated_result(output_file):
            return (True, f"成功将 {len(krn_files)} 个编码文件整合并保存到 {output_file}")
        else:
            return (False, "整合成功，但保存结果失败")

    except Exception as e:
        return (False, f"整合过程出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    KRN_OUTPUT_DIR = "/tmp/pycharm_project_588/grand_staff_omr/SMT-master/Data/kern_out"  # 模型输出的编码文件所在目录
    INTEGRATED_OUTPUT = "/tmp/pycharm_project_588/grand_staff_omr/SMT-master/Data/Fulloutcomes"  # 整合结果保存路径

    # 执行整合
    success, message = process_krn_integration(
        krn_dir=KRN_OUTPUT_DIR,
        output_file=INTEGRATED_OUTPUT
    )

    print(message)

