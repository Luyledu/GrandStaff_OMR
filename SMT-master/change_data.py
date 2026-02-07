import os
from datasets import Dataset
from PIL import Image


def load_music_labels(label_dir, label_ext=".bekrn"):
    """
    读取乐谱标签文件（支持bekern或kern格式）
    :param label_dir: 标签文件夹路径
    :param label_ext: 标签文件扩展名（.bekern 或 .kern）
    :return: {样本ID: 标签内容} 的字典
    """
    label_dict = {}
    for filename in os.listdir(label_dir):
        if filename.endswith(label_ext):
            sample_id = os.path.splitext(filename)[0]  # 提取文件名作为样本ID
            with open(os.path.join(label_dir, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                # 标准化处理：替换换行符为特殊标记（避免与模型预处理冲突）
                # 保留乐谱格式的结构性换行（用<b>标记），空格用<s>标记（后续模型会解析）
                content = content.replace("\n", " <b> ").replace("\r", "").replace("  ", " <s> ")
                label_dict[sample_id] = content
    return label_dict


def merge_labels(bekern_dict, kern_dict=None):
    """
    可选：合并bekern和kern标签（若需要补充信息）
    :param bekern_dict: bekern标签字典
    :param kern_dict: kern标签字典（可选）
    :return: 合并后的标签字典
    """
    merged_dict = {}
    for sample_id, bekern_content in bekern_dict.items():
        if kern_dict and sample_id in kern_dict:
            # 合并格式：[KERN] kern内容 [BEKERN] bekern内容
            merged_content = f"[KERN] {kern_dict[sample_id]} [BEKRN] {bekern_content}"
            merged_dict[sample_id] = merged_content
        else:
            merged_dict[sample_id] = bekern_content
    return merged_dict


def create_music_dataset(image_dir, bekern_dir, kern_dir=None, save_path=None, use_kern=False):
    """
    创建包含image和transcription字段的乐谱数据集
    :param image_dir: JPG图像文件夹路径
    :param bekern_dir: bekern标签文件夹路径
    :param kern_dir: kern标签文件夹路径（可选）
    :param save_path: 数据集保存路径（可选）
    :param use_kern: 是否合并kern标签（默认False）
    :return: HuggingFace Dataset对象
    """
    # 1. 读取标签（优先读取bekern）
    bekern_dict = load_music_labels(bekern_dir, label_ext=".bekrn")
    # 可选：读取kern标签并合并
    if use_kern and kern_dir:
        kern_dict = load_music_labels(kern_dir, label_ext=".krn")
        label_dict = merge_labels(bekern_dict, kern_dict)
    else:
        label_dict = bekern_dict

    # 2. 关联图像与标签
    samples = []
    for sample_id, transcription in label_dict.items():
        # 查找对应的JPG图像
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        if not os.path.exists(image_path):
            print(f"警告：未找到 {sample_id} 对应的图像，跳过")
            continue

        # 验证图像有效性
        try:
            with Image.open(image_path) as img:
                img.verify()  # 检查图像完整性
            samples.append({
                "image": image_path,  # 图像路径（训练时可加载为PIL图像）
                "transcription": transcription  # 转换后的标签字段
            })
        except Exception as e:
            print(f"跳过损坏的图像 {image_path}：{e}")

    # 3. 转换为Dataset格式
    if not samples:
        raise ValueError("未生成任何有效样本，请检查文件路径和文件名匹配")
    dataset = Dataset.from_list(samples)

    # 4. 保存数据集
    if save_path:
        dataset.save_to_disk(save_path)
        print(f"数据集已保存至 {save_path}，共 {len(dataset)} 个样本")

    # 打印示例
    print("样本示例：")
    print("图像路径：", dataset[0]["image"])
    print("转录标签：", dataset[0]["transcription"][:100] + "...")  # 显示前100字符

    return dataset


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    IMAGE_DIR = "/tmp/pycharm_project_588/datasets/SMTyuandata/fullgrandstaff/imgaes"  # JPG图像文件夹
    BEKERN_DIR = "/tmp/pycharm_project_588/datasets/SMTyuandata/fullgrandstaff/transcription"  # bekern标签文件夹
    KERN_DIR = "raw_dataset/kern_labels"  # kern标签文件夹（可选）
    SAVE_PATH = "/tmp/pycharm_project_588/datasets/SMTyuandata/train_G/train"  # 保存路径
    # 执行转换（use_kern=True 表示合并kern标签，False则仅用bekern）
    dataset = create_music_dataset(
        image_dir=IMAGE_DIR,
        bekern_dir=BEKERN_DIR,
        kern_dir=KERN_DIR,
        save_path=SAVE_PATH,
        use_kern=False  # 根据需求选择是否合并kern
    )
