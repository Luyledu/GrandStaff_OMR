# import torch
# import cv2
# from data_augmentation.data_augmentation import convert_img_to_tensor
# from smt_model import SMTModelForCausalLM
#
# image = cv2.imread(r"/tmp/pycharm_project_588/omr.model/SMT-master/Data/image256")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SMTModelForCausalLM.from_pretrained(r"/tmp/pycharm_project_588/omr.model/SMT-master/model/smt-grandstaff").to(device)
#
# predictions, _ = model.predict(convert_img_to_tensor(image).unsqueeze(0).to(device),
#                                convert_to_str=True)
#
# print("".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t'))
#修改过后的程序，支持处理单张或多张照片，结果可保存到指定文件夹
import os
import torch
import cv2
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM


def process_image(image_path, model, device):
    """处理单张图片并返回识别结果"""
    print(f"正在读取图像: {image_path}")
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    print(f"图像尺寸: {image.shape}")

    # 模型推理
    with torch.no_grad():
        input_tensor = convert_img_to_tensor(image).unsqueeze(0).to(device)
        print(f"输入张量形状: {input_tensor.shape}")
        predictions, _ = model.predict(input_tensor, convert_to_str=True)

    # 检查预测结果
    if not predictions:
        print("警告: 模型预测结果为空")
        return ""

    # 处理结果
    result = "".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t')
    print(f"识别结果长度: {len(result)}")
    return result


def save_result(result, output_path, base_name):
    """保存识别结果为krn格式文件"""
    # 检查结果是否为空
    if not result.strip():
        print("警告: 识别结果为空，不会保存文件")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    print(f"输出目录: {output_path}")

    # 生成输出文件名（使用原文件名，替换扩展名为.krn）
    file_name = os.path.splitext(base_name)[0] + ".krn"
    output_file = os.path.join(output_path, file_name)
    print(f"完整保存路径: {output_file}")

    # 添加krn文件头（如果需要）
    # 这里假设result已经是正确的krn格式，如果需要可以添加必要的头信息
    krn_content = result

    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(krn_content)
        print(f"结果已成功保存为krn格式: {output_file}")

        # 验证文件是否真的存在
        if os.path.exists(output_file):
            print(f"验证: 文件确实存在")
            print(f"文件大小: {os.path.getsize(output_file)} 字节")
        else:
            print("错误: 文件保存后未找到，可能是权限问题")

    except Exception as e:
        print(f"保存文件时出错: {str(e)}")


def main():
    # 在这里直接设置文件路径
    INPUT_PATH = r"/tmp/pycharm_project_588/YOLO11/score_cut/zhangjie"  # 输入图像或图像文件夹
    OUTPUT_PATH = r"/tmp/pycharm_project_588/omr.model/SMT-master/Data/kern_out"  # 输出结果文件夹
    MODEL_PATH = r"/tmp/pycharm_project_588/omr.model/SMT-master/model/smt-grandstaff"  # 模型路径
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']  # 支持的图像扩展名

    try:
        # 准备设备和模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        model = SMTModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()

        # 获取要处理的图像列表
        image_paths = []

        if os.path.isfile(INPUT_PATH):
            # 处理单张图片
            if os.path.splitext(INPUT_PATH)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(INPUT_PATH)
            else:
                raise ValueError(f"输入文件不是有效图像: {INPUT_PATH}")
        elif os.path.isdir(INPUT_PATH):
            # 处理文件夹中的所有图片
            for root, _, files in os.walk(INPUT_PATH):
                for file in files:
                    if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                        image_paths.append(os.path.join(root, file))
            if not image_paths:
                raise ValueError(f"在目录中未找到有效图像: {INPUT_PATH}")
        else:
            raise ValueError(f"输入路径不存在: {INPUT_PATH}")

        print(f"找到 {len(image_paths)} 个图像文件")

        # 处理每张图片
        for image_path in image_paths:
            print(f"\n=== 处理图像: {image_path} ===")
            base_name = os.path.basename(image_path)

            try:
                # 处理图像
                result = process_image(image_path, model, device)

                # 保存结果为krn格式
                save_result(result, OUTPUT_PATH, base_name)

            except Exception as e:
                print(f"处理图像 {base_name} 时出错: {str(e)}")
                print("继续处理下一张图像...")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()