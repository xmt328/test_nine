import os


from paddlex.utils.result_saver import try_except_decorator
from paddlex.utils.config import parse_args, get_config
from paddlex.utils.errors import raise_unsupported_api_error
from paddlex.model import _ModelBasedConfig

print(f"""数据集格式如下：
      dataset
      ├─images    #所有图片存放路径
      ├─label.txt #标签路径，每一行数据格式为 <序号>+<空格>+<类别>，如15 地球仪
      ├─train.txt #训练图片，每一行数据格式为 <图片路径>+<空格>+<类别>，如images/001.jpg 0
      └─验证集和测试集同上
      """)
class Engine(object):
    """Engine"""

    def __init__(self):
        args = parse_args()
        args.config='PP-HGNetV2-B4.yaml'
        args.override=['Global.mode=train', 'Global.dataset_dir=dataset']
        config = get_config(args.config, overrides=args.override, show=False)
        self._mode = config.Global.mode
        self._output = config.Global.output
        self._model = _ModelBasedConfig(config)

    @try_except_decorator
    def run(self):
        """the main function"""
        if self._mode == "check_dataset":
            return self._model.check_dataset()
        elif self._mode == "train":
            self._model.train()
        elif self._mode == "evaluate":
            return self._model.evaluate()
        elif self._mode == "export":
            return self._model.export()
        elif self._mode == "predict":
            for res in self._model.predict():
                res.print(json_format=False)
        else:
            raise_unsupported_api_error(f"{self._mode}", self.__class__)

if __name__ == "__main__":
    Engine().run()