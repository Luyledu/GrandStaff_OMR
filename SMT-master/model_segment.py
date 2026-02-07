from PIL import Image, ImageOps
import os


def detect_braces(binary_img, min_width=10):
    """检测大括号位置"""
    width, height = binary_img.size
    pixels = binary_img.load()
    brace_positions = []

    for y in range(height):
        # 扫描每行的黑色像素段
        start = None
        for x in range(width):
            if pixels[x, y] == 0:  # 黑色像素
                if start is None:
                    start = x
            else:
                if start is not None:
                    if x - start >= min_width:
                        brace_positions.append((start, x - 1, y))
                    start = None
        # 处理行末尾的线段
        if start is not None:
            brace_positions.append((start, width - 1, y))

    # 过滤垂直方向连续的线段
    valid_braces = []
    prev_x = -1
    for x1, x2, y in brace_positions:
        if x1 <= prev_x + 2:  # 合并相邻线段
            valid_braces[-1] = (min(x1, valid_braces[-1][0]), x2, y)
        else:
            valid_braces.append((x1, x2, y))
        prev_x = x1

    # 返回中间位置
    return [(x1 + x2) // 2 for x1, x2, y in valid_braces]


def split_staff(image_path, output_dir='output'):
    """主分割函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 图像预处理
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    binary = img.point(lambda x: 0 if x < 128 else 255, '1')

    # 检测大括号
    brace_pos = detect_braces(binary)

    # 执行分割
    if not brace_pos:
        print("未检测到大括号，保存完整乐谱")
        img.save(f"{output_dir}/full_staff.png")
        return

    # 去重排序
    brace_pos = sorted(list(set(brace_pos)))

    # 执行裁剪
    parts = []
    last = 0
    width = img.width

    for pos in brace_pos:
        left = max(last, 0)
        right = min(pos, width)
        parts.append(img.crop((left, 0, right, img.height)))
        last = right

    # 处理最后部分
    parts.append(img.crop((last, 0, width, img.height)))

    # 保存结果
    for i, part in enumerate(parts):
        part.save(f"{output_dir}/part_{i}.png")
    print(f"分割完成，保存为 {len(parts)} 个单元")


# 使用示例
if __name__ == "__main__":
    input_path = "E:\keyan\YOLO11/runs\detect\predict5\牧歌_01.jpg"  # 替换为你的五线谱图片
    split_staff(input_path)