import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("runs/train/exp-seg-goldyolo-use/weights/best.pt", task="detect")

# 打开相机
cap = cv2.VideoCapture(0)  # 使用默认相机，如果有多个相机，可能需要更改索引

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧大小
    frame = cv2.resize(frame, (640, 640))

    # 进行目标检测
    results = model(frame)

    # 在帧上绘制检测结果
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # 获取检测框坐标
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 计算中心点坐标
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            print(center_x, center_y)

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # 在中心点旁边标注坐标
            cv2.putText(frame, f'({center_x}, {center_y})', (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 显示结果
    cv2.imshow('Detection with Center Points', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()