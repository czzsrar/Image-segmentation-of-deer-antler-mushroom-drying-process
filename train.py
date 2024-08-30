import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # 加载 YOLO 模型
    model = YOLO('yolov8-seg.yaml')

    # 训练模型
    results = model.train(
        data='mogu_new.yaml',
        epochs=250,
        batch=-1,
        workers=0,
        project='runs/train',
        name='exp'
    )
