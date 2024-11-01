# 九宫格测试代码

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## 参考项目

模型及V4数据集：https://github.com/taisuii/ClassificationCaptchaOcr

api：https://github.com/ravizhan/geetest-v3-click-crack

## 运行步骤

### 1.安装依赖

如果要训练paddle的话还得安装paddlex及图像分类模块，安装看项目https://github.com/PaddlePaddle/PaddleX
模型需要新建一个model文件夹，然后放进去，具体命名可以是resnet18.onnx或者PP-HGNetV2-B4.onnx

```
pip install -r requirements.txt
```

### 2.自行准备数据集，V3和V4有区别

##### a. 训练resnet18

- 数据集详情参考上面标注的项目，但是上面项目是V4数据集，V3没有demo，自行发挥吧，用V4练V3不改代码正确率有点感人
- 主要是V4的尺寸和V3有差别，V4的api直接给两张图，一张是目标图，一张是九宫格，V3放在一起要切目标，且V3目标图清晰度很低，V4九宫格切了之后是100 * 86的图（去掉黑边），但是V3九宫格切的是112 * 112，不确定V4九宫格内容在V3基础上做了什么变换，反正改预处理就完事了

##### b. 训练PP-HGNetV2-B4

在paddle上随便找的，数据集格式如下，如果拿V4练V3，建议是多整点变换

```
   dataset
   ├─images   #所有图片存放路径
   ├─label.txt #标签路径，每一行数据格式为 <序号>+<空格>+<类别>，如15 地球仪
   ├─train.txt #训练图片，每一行数据格式为 <图片路径>+<空格>+<类别>，如images/001.jpg 0
   └─验证集和测试集同上
```

##### c. 如果要切V3的图用crop_image.py的crop_image_v3，切V4则使用crop_image，自行编写切图脚本

### 3.训练模型

- 训练resnet18运行 `python train.py`
- 如果训练PP-HGNetV2-B4运行`python train_paddle.py`

### 4.模型转换为onnx

- 运行 `python convert.py`（自行进去修改需要转换的模型，一般是选loss小的）
- paddle模型转换要装paddle2onnx，详情参见https://www.paddlepaddle.org.cn/documentation/docs/guides/advanced/model_to_onnx_cn.html

### 5.启动fastapi服务

运行 `python main.py`（默认用的paddle的onnx模型，如果要用resnet18可以自己改注释）

由于轨迹问题，可能会出现验证正确但是结果失败，所以建议增加retry次数

### 6.api调用

python调用如：

```python
import httpx

def game_captcha(gt: str, challenge: str):
	res = httpx.get("http://127.0.0.1:9645/pass_nine",params={'gt':gt,'challenge':challenge,'use_v3_model':True},timeout=10)
	datas = res.json()['data']
    if datas['result'] == 'success':
        return datas['validate']
    return None # 失败返回None 成功返回validate
```

#### --宣传--

欢迎大家支持我的其他项目喵~~~~~~~~
