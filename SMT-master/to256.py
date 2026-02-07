#调整照片宽度
# import cv2
# import numpy as np
#
# def resize_image_width_cv(input_path, output_path, target_width=256):
#     try:
#         # 读取图片（自动处理多种格式）
#         img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             raise ValueError("无法读取图片文件")
#
#         # 计算新尺寸
#         h, w = img.shape[:2]
#         new_height = int(h * (target_width / w))
#
#         # 调整尺寸
#         resized = cv2.resize(img, (target_width, new_height),
#                            interpolation=cv2.INTER_AREA)
#
#         # 保存图片（自动推断格式）
#         cv2.imwrite(output_path, resized)
#         print(f"成功处理：{input_path} -> {output_path}")
#
#     except Exception as e:
#         print(f"处理失败 {input_path}: {str(e)}")
#         print("详细错误信息:")
#
#
# # 使用示例
# resize_image_width_cv(r"E:\keyan\omr.model\SMT-master\Data\兰亭序.png",
#                     r"E:\keyan\omr.model\SMT-master\Data\output.png")

#调整照片高度
# import os
# import imghdr
# import cv2
# import numpy as np
# import traceback
# from PIL import Image
# import sys
# import locale
#
#
# # === 辅助函数区 ===
# def verify_image_format(file_path):
#     """验证图片格式的两种方法"""
#     # 方法1：通过文件扩展名验证
#     ext = os.path.splitext(file_path)[1].lower()
#     if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
#         return f"扩展名不匹配: {ext}"
#
#     # 方法2：通过文件头验证（更可靠）
#     try:
#         with open(file_path, 'rb') as f:
#             file_header = f.read(12)
#             actual_format = imghdr.what(None, file_header)
#             if actual_format not in ['png', 'jpeg', 'bmp', 'tiff']:
#                 return f"文件头格式不匹配: {actual_format}"
#     except Exception as e:
#         return f"文件读取失败: {str(e)}"
#
#     return "格式验证通过"
#
#
# def fix_path_encoding(path):
#     """处理Windows系统中文路径编码问题"""
#     if sys.platform.startswith('win'):
#         encoding = locale.getpreferredencoding()
#         try:
#             return path.encode(encoding).decode('gbk')
#         except:
#             pass
#     return path
#
#
# # === 主处理函数 ===
# def resize_image_height(input_path, output_path, target_height=256):
#     try:
#         # 路径编码修正
#         input_path = fix_path_encoding(input_path)
#         output_path = fix_path_encoding(output_path)
#
#         # 格式验证
#         check_result = verify_image_format(input_path)
#         if check_result != "格式验证通过":
#             raise ValueError(check_result)
#
#         # 尝试Pillow方案
#         try:
#             with Image.open(input_path) as img:
#                 # 计算新尺寸
#                 width, height = img.size
#                 if height == 0:
#                     raise ValueError("无效图片高度")
#                 new_width = int(width * (target_height / height))
#
#                 # 调整尺寸
#                 resized = img.resize(
#                     (new_width, target_height),
#                     Image.Resampling.LANCZOS
#                 )
#
#                 # 保存图片（自动处理格式）
#                 format = os.path.splitext(output_path)[1][1:].upper()
#                 if format == 'JPG':
#                     format = 'JPEG'
#                 resized.save(output_path, format=format)
#                 return
#
#         except Exception as e:
#             print(f"Pillow处理失败，尝试OpenCV方案: {str(e)}")
#
#         # OpenCV备用方案（更强大的格式支持）
#         try:
#             # 读取图片（自动处理多种格式）
#             img = cv2.imdecode(
#                 np.fromfile(input_path, dtype=np.uint8),
#                 cv2.IMREAD_UNCHANGED
#             )
#             if img is None:
#                 raise ValueError("无法读取图片文件")
#
#             # 计算新尺寸
#             h, w = img.shape[:2]
#             if h == 0:
#                 raise ValueError("无效图片高度")
#             new_width = int(w * (target_height / h))
#
#             # 调整尺寸
#             resized = cv2.resize(
#                 img,
#                 (new_width, target_height),
#                 interpolation=cv2.INTER_AREA
#             )
#
#             # 保存图片（自动处理中文路径）
#             cv2.imencode(
#                 os.path.splitext(output_path)[1][1:],
#                 resized
#             )[1].tofile(output_path)
#
#         except Exception as e:
#             raise RuntimeError(f"双方案均处理失败: {str(e)}")
#
#     except Exception as e:
#         print(f"处理失败 {input_path}: {str(e)}")
#         traceback.print_exc()
#         raise
#
#
# # === 使用示例 ===
# if __name__ == "__main__":
#     # 测试单文件处理
#     try:
#         resize_image_height(
#             r"/tmp/pycharm_project_588/omr.model/SMT-master/Data/yuanimages/kmxzbpntv5o(1).png",
#             r"/tmp/pycharm_project_588/omr.model/SMT-master/Data/image256"
#         )
#         print("处理成功！")
#     except Exception as e:
#         print(f"测试失败: {str(e)}")
import os
import argparse
from PIL import Image


def resize_image(input_path, output_path, target_height=256):
    """调整单张图片尺寸"""
    try:
        with Image.open(input_path) as img:
            width, height = img.size
            new_width = int(width * (target_height / height))
            resized_img = img.resize((new_width, target_height), Image.LANCZOS)
            resized_img.save(output_path)
            print(f"已调整: {input_path} -> {output_path}")
    except Exception as e:
        print(f"处理失败: {input_path} - {str(e)}")


def process_files(input_path, output_path, target_height=256):
    """处理单张图片或目录中的所有图片"""
    if os.path.isfile(input_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resize_image(input_path, output_path, target_height)
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)
            if os.path.isfile(input_file):
                try:
                    with Image.open(input_file) as _:
                        output_file = os.path.join(output_path, filename)
                        resize_image(input_file, output_file, target_height)
                except (IOError, OSError):
                    print(f"跳过非图片文件: {input_file}")
    else:
        print(f"错误: 路径不存在 - {input_path}")


def main():
    """主函数：直接在代码中设置参数"""
    # 直接在这里修改路径和高度参数
    config = {
        'input_path': 'E:\keyan\omr.model\SMT-master\Data\yuanimages',  # 单张图片路径或目录
        'output_path': 'E:\keyan\omr.model\SMT-master\Data\image256',  # 输出路径或目录
        'target_height': 256  # 目标高度（像素）
    }

    process_files(
        config['input_path'],
        config['output_path'],
        config['target_height']
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
