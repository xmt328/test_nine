from train import MyResNet18
import torch

def convert():
    # 加载 PyTorch 模型
    model_path = "model/resnet18_39_0.01445627337038193.pth"
    model = MyResNet18(num_classes=91)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 生成一个示例输入
    dummy_input = torch.randn(10, 3, 224, 224)
    # 将模型转换为 ONNX 格式
    torch.onnx.export(model, dummy_input, "model/resnet18.onnx", verbose=True)


if __name__ == '__main__':
    convert()