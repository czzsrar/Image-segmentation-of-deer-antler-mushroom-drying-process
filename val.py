import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp-seg-C2f-Faster-EMA-use/weights/best.pt')
    model.val(data='mogu_new.yaml',
              split='val',
            #   imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )