import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLO 模型
    model = YOLO('yolov8-seg-goldyolo.yaml')

    # 训练模型
    results = model.train(
        data='structure.yaml',
        epochs=5,
        batch=-1,
        workers=0,
        degrees=45, # (float) 图像旋转角度
        flipud=0.5, # (float) 上下翻转概率
        hsv_h=0.015,  # (float) 色调变换
        hsv_s=0.7, # (float) 饱和度变换
        hsv_v=0.4, # (float) 亮度变换
        project='runs/train',
        name='exp'
    )

