import os

import numpy as np


from .crop_image import crop_image, convert_png_to_jpg,draw_points_on_image

import time
import cv2
from PIL import Image
from io import BytesIO
import onnxruntime as ort


def predict(icon_image, bg_image):
    from train import MyResNet18, data_transform
    import torch
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'resnet18_38_0.021147585306924.pth')
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    target_images = []
    target_images.append(data_transform(Image.open(BytesIO(icon_image))))

    bg_images = crop_image(bg_image, coordinates)
    for bg_image in bg_images:
        target_images.append(data_transform(bg_image))

    start = time.time()
    model = MyResNet18(num_classes=91)  # 这里的类别数要与训练时一致
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("加载模型，耗时:", time.time() - start)
    start = time.time()

    target_images = torch.stack(target_images, dim=0)
    target_outputs = model(target_images)

    scores = []

    for i, out_put in enumerate(target_outputs):
        if i == 0:
            # 增加维度，以便于计算
            target_output = out_put.unsqueeze(0)
        else:
            similarity = torch.nn.functional.cosine_similarity(
                target_output, out_put.unsqueeze(0)
            )
            scores.append(similarity.cpu().item())
    # 从左到右，从上到下，依次为每张图片的置信度
    print(scores)
    # 对数组进行排序，保持下标
    indexed_arr = list(enumerate(scores))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # 提取最大三个数及其下标
    largest_three = sorted_arr[:3]
    print(largest_three)
    print("识别完成，耗时:", time.time() - start)

def load_model(name='PP-HGNetV2-B4.onnx'):
    # 加载onnx模型
    global session,input_name
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print("加载模型，耗时:", time.time() - start)


def predict_onnx(icon_image, bg_image, point = None):
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]

    def cosine_similarity(vec1, vec2):
        # 将输入转换为 NumPy 数组
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        # 计算向量的范数
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        # 计算余弦相似度
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def data_transforms(image):
        image = image.resize((224, 224))
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB))
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)
        return image_array

    target_images = []
    target_images.append(data_transforms(Image.open(BytesIO(icon_image))))
    bg_images = crop_image(bg_image, coordinates)

    for one in bg_images:
        target_images.append(data_transforms(one))

    start = time.time()
    outputs = session.run(None, {input_name: target_images})[0]

    scores = []
    for i, out_put in enumerate(outputs):
        if i == 0:
            target_output = out_put
        else:
            similarity = cosine_similarity(target_output, out_put)
            scores.append(similarity)
    # 从左到右，从上到下，依次为每张图片的置信度
    # print(scores)
    # 对数组进行排序，保持下标
    indexed_arr = list(enumerate(scores))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # 提取最大三个数及其下标
    if point == None:
        largest_three = sorted_arr[:3]
        answer = [coordinates[i[0]] for i in largest_three]
    # 基于分数判断
    else:
        answer = [one[0] for one in sorted_arr if one[1] > point]
    print(f"识别完成{answer}，耗时: {time.time() - start}")
    draw_points_on_image(bg_image, answer)
    return answer

def predict_onnx_pdl(images_path):
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    def data_transforms(path):
        # 打开图片
        img = Image.open(path)
        # 调整图片大小为232x224（假设最短边长度调整为232像素）
        if img.width < img.height:
            new_size = (232, int(232 * img.height / img.width))
        else:
            new_size = (int(232 * img.width / img.height), 232)
        resized_img = img.resize(new_size, Image.BICUBIC)
        # 裁剪图片为224x224
        cropped_img = resized_img.crop((0, 0, 224, 224))
        # 将图像转换为NumPy数组并进行归一化处理
        img_array = np.array(cropped_img).astype(np.float32)
        img_array /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_array -= np.array(mean)
        img_array /= np.array(std)
        # 将通道维度移到前面
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
    images = []
    for pic in sorted(os.listdir(images_path)):
        if "cropped" not in pic:
            continue
        image_path = os.path.join(images_path,pic)
        images.append(data_transforms(image_path))
    if len(images) == 0:
        raise FileNotFoundError(f"先使用切图代码切图至{image_path}再推理,图片命名如cropped_9.jpg,从0到9共十个,最后一个是检测目标")
    start = time.time()
    outputs = session.run(None, {input_name: images})[0]
    result = [np.argmax(one) for one in outputs]
    target = result[-1]
    answer = [coordinates[index] for index in range(9) if result[index] == target]
    print(f"识别完成{answer}，耗时: {time.time() - start}")
    if os.path.exists(os.path.join(images_path,"nine.jpg")):
        with open(os.path.join(images_path,"nine.jpg"),'rb') as f:
            bg_image = f.read()
        draw_points_on_image(bg_image, answer)
    return answer
    
    

if __name__ == "__main__":
    # 使用resnet18.onnx
    # load_model("resnet18.onnx")
    # icon_path = "img_2_val/cropped_9.jpg"
    # bg_path = "img_2_val/nine.jpg"
    # with open(icon_path, "rb") as rb:
    #     if icon_path.endswith('.png'):
    #         icon_image = convert_png_to_jpg(rb.read())
    #     else:
    #         icon_image = rb.read()
    # with open(bg_path, "rb") as rb:
    #     bg_image = rb.read()
    # predict_onnx(icon_image, bg_image)
    
    # 使用PP-HGNetV2-B4.onnx
    load_model()
    predict_onnx_pdl(r'img_saved\img_fail\7fe559a85bac4c03bc6ea7b2e85325bf')
else:
    load_model()