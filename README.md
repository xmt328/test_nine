# 九宫格测试代码

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## **本项目仅供学习交流使用，请勿用于商业用途，否则后果自负。**

## 参考项目

模型及V4数据集：https://github.com/taisuii/ClassificationCaptchaOcr

api：https://github.com/ravizhan/geetest-v3-click-crack

## 运行步骤

### 1.安装依赖

```
pip install -r requirements.txt
```

### 2.自行准备数据集，V3和V4有区别

- 数据集详情参考上面标注的项目，但是上面项目是V4数据集，V3没有demo，自行发挥吧，用V4正确率有点感人，或许可以试试别的模型看看能不能泛化

- 如果要切V3的图用crop_image.py的crop_image_v3，切V4则使用crop_image，自行编写切图脚本


### 3.训练模型

训练运行 `python train.py`

### 4.模型转换为onnx

运行 `python convert.py`（自行进去修改需要转换的模型，一般是选loss小的）

### 5.启动fastapi服务

运行 `python main.py`

### 6.api调用

python调用如：

```python
import httpx

res = httpx.get("http://127.0.0.1:9645/pass_nine",params={'gt':gt,'challenge':challenge},timeout=10)

datas = res.json()['data']

if datas['result'] == 'success':

	return datas['validate']
```







