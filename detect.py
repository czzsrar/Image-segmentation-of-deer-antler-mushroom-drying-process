import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('F:/yolo/ultralytics-20240513/ultralytics-main/runs/train/exp-seg-use/weights/best.pt') # select your model.pt path
    model = YOLO('F:/yolo/ultralytics-20240513/ultralytics-main/runs/train/exp-seg-C2f-Faster-EMA-sppf-eiou-x/weights/best.pt')
    model.predict(source='F:/yolo/ultralytics-20240513/ultralytics-main/datasets/mogu_labels_2/images/train/2_20240529_160804.jpg',
                  # show = True,
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  conf=0.8,
                  # visualize=True # visualize model features maps
                )