import numpy as np
import cv2
from scipy import stats
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
from ultralytics import YOLO
from scipy.interpolate import splprep, splev
from scipy.integrate import quad

from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import Config
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import OBError
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image

# 初始化YOLO模型
yolo = YOLO("YOLOv8n-seg.pt", task="detect")

def extract_feature_points(mask, interval=20, min_width=10):
    h, w = mask.shape
    points = []
    for y in range(0, h, interval):
        row = mask[y, :]
        left = np.argmax(row)
        right = w - np.argmax(row[::-1]) - 1
        if right - left > min_width:
            x = (left + right) // 2
            points.append((x, y))
    return np.array(points)

def fit_line(points):
    if len(points) < 2:
        return None, None
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept

def select_main_navigation_line(lines):
    if not lines:
        return None
    center_x, center_y = 320, 640
    distances = []
    for slope, intercept in lines:
        y = slope * center_x + intercept
        distance = abs(y - center_y)
        distances.append(distance)
    return lines[np.argmin(distances)]

def smooth_parameters(slope, intercept):
    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.x = np.array([slope, intercept])
    kf.F = np.eye(2)
    kf.H = np.eye(2)
    kf.P *= 1000.
    kf.R = np.eye(2) * 0.1
    kf.Q = np.eye(2) * 0.001

    for _ in range(10):
        kf.predict()
        kf.update(np.array([slope, intercept]))

    return kf.x[0], kf.x[1]

def fit_curve_and_get_length(points):
    points = points[points[:, 1].argsort()]
    tck, _ = splprep([points[:, 0], points[:, 1]], s=0, k=3)
    t = np.linspace(0, 1, 1000)
    smooth_points = np.array(splev(t, tck)).T
    
    def curve_length_integrand(t):
        dx_dt, dy_dt = splev(t, tck, der=1)
        return np.sqrt(dx_dt**2 + dy_dt**2)
    
    length, _ = quad(curve_length_integrand, 0, 1)
    
    return smooth_points, length

def draw_curve(image, points, mask, color=(0, 255, 255), thickness=2):
    for i in range(len(points) - 1):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[i+1].astype(int))
        if mask[pt1[1], pt1[0]] > 0 and mask[pt2[1], pt2[0]] > 0:
            cv2.line(image, pt1, pt2, color, thickness)

def process_image(source):
    result = yolo(source, save=True)

    if result[0].masks is not None and len(result[0].masks) > 0:
        masks_data = result[0].masks.data
        all_lines = []
        
        for mask in masks_data:
            mask = (mask.cpu().numpy() * 255).astype(np.uint8)
            
            feature_points = extract_feature_points(mask)
            if len(feature_points) > 1:
                smooth_points, curve_length = fit_curve_and_get_length(feature_points)
                draw_curve(source, smooth_points, mask)
                
                print(f"Curve length: {curve_length:.2f} pixels")
                
                slope, intercept = fit_line(feature_points)
                if slope is not None and intercept is not None:
                    all_lines.append((slope, intercept))
                
                for point in feature_points:
                    if mask[point[1], point[0]] > 0:
                        cv2.circle(source, tuple(point), 2, (0, 255, 0), -1)
        
        main_line = select_main_navigation_line(all_lines)
        
        if main_line is not None:
            smoothed_line = smooth_parameters(main_line[0], main_line[1])
            
            center_x = 320
            center_y = 640
            angle = np.arctan(smoothed_line[0]) * 180 / np.pi
            heading_error = angle
            lateral_error = abs(center_y - (smoothed_line[0] * center_x + smoothed_line[1]))
            
            print(f"Heading Error: {heading_error:.2f} degrees")
            print(f"Lateral Error: {lateral_error:.2f} pixels")
    
    return source

def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            
            processed_image = process_image(color_image)
            
            cv2.imshow("Processed Image", processed_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 27 is ESC key
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()

if __name__ == "__main__":
    main()