import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp-seg-goldyolo-asf/weights/best.pt') # select your model.pt path
    model.track(source='mogu',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )