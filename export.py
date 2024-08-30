import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu
# http://localhost:6006/
if __name__ == '__main__':
    model = YOLO('yolov8-seg-goldyolo.yaml')
    model.export(format='onnx', simplify=True, opset=13)