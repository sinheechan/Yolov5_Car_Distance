# 객체 거리 측정 : Yolov5

import torch
import numpy as np
import yolov5
import cv2
import math
from time import time

class DistanceEstimationDetector:
    def __init__(self, video_path, model_path):
        """
        :param video_path: 처리할 영상
        :param model_path: 모델
        """
        self.video_path = video_path
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        :return: 비디오 리턴
        """
        return cv2.VideoCapture(self.video_path)

    def load_model(self, model_path):
        """
        Model 로드 및 구성
        :param model_path: 로드헬 모델 파일 경로
        :return: 로드 및 구성된 모델
        """
        model = yolov5.load(model_path)
        model.conf = 0.40  # confidence 임계값
        model.iou = 0.45  # NMS IoU threshold
        model.max_det = 1000  # 최대 감지 수(한 프레임당)
        model.classes = [0, 2]  # 객체의 class number
        return model

    def get_model_results(self, frame):
        """
        예측 및 예측 결과 반환
        :param frame: 동영상 프레임
        :return: 입력 프레임의 결과
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame,size=640)
        predictions = results.xyxyn[0]
        cords, scores, labels = predictions[:, :4], predictions[:, 4], predictions[:,5]
        return cords, scores, labels

    def draw_rect(self, results, frame):
        """
        바운딩 박스 그리기
        :param results: 모델에서 반환된 객체 감지 결과
        :param frame: 처리중인 프레임
        :return: 바운딩 박스처리된 이미지
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        n = len(labels)  # 감지된 개체(인스턴스) 수

        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            green_bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), green_bgr, 1) # 바운딩 박스 그리기
            cls = labels[i] # 클래스 네임 얻기
            cls = int(cls)
            global cls_name
            if cls == 2:
                cls_name = 'car'
            if cls == 0:
                cls_name = 'person'
            cv2.putText(frame, cls_name, (x1+35, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 1)

        return frame
    
    def calc_distances(self, results, frame):
        """
        거리 계산
        :param results: 객체 감지로 얻은 결과 값
        :param frame: 처리할 프레임
        :return: 거리 계산을 수행하고 처리한 이미지
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        points = []

        for car in cord:
            x1, y1, x2, y2 = int(car[0] * x_shape), int(car[1] * y_shape), int(car[2] * x_shape), int(
                car[3] * y_shape)  # 위치
            x_mid_rect, y_mid_rect = (x1 + x2) / 2, (y1 + y2) / 2  # 축의 중간점
            y_line_length, x_line_length = abs(y1 - y2), abs(x1 - x2)  # 축의 길이
          #  cv2.circle(frame, center=(int(x_mid_rect), int(y_mid_rect)), radius=1, color=(0, 0, 255), thickness=5)
            points.append([x1, y1, x2, y2, int(x_mid_rect), int(y_mid_rect), int(x_line_length), int(y_line_length)])

        x_shape_mid = int(x_shape / 2)
        start_x, start_y = x_shape_mid, y_shape
        start_point = (start_x, start_y)

        heigth_in_rf = 121
        measured_distance = 275  # inch = 700cm
        real_heigth = 60  # inch = 150 cm
        focal_length = (heigth_in_rf * measured_distance) / real_heigth

        pixel_per_cm = float(2200 / x_shape) * 2.54

        for i in range(0, len(points)):
            end_x1, end_y1, end_x2, end_y2, end_x_mid_rect, end_y_mid_rect, end_x_line_length, end_y_line_length = points[i]
            if end_x2 < x_shape_mid: # 왼쪽에 있는 경우
                end_point = (end_x2, end_y2) # 오른쪽 하단 선택
            elif end_x1 > x_shape_mid: # 오른쪽에 있는 경우
                end_point = (end_x1, end_y2) # 왼쪽 하단 선택
            else: # 중앙에 있는 경우
                end_point = (end_x_mid_rect,end_y2) # 밑 중간 선택

            dif_x, dif_y = abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1])
            pixel_count = math.sqrt(math.pow(dif_x, 2) + math.pow(dif_y, 2))
            global distance
            distance = float(pixel_count * pixel_per_cm  / end_y_line_length)

            # distance = real_heigth * focal_length / abs(end_y1 - end_y2);
            # distance = distance * 2.54 / 100
            #  print(distance)
            #cv2.line(frame, start_point, end_point, color=(0, 0, 255), thickness=1)
            cv2.putText(frame, str(round(distance, 2)) +" m", (int(end_x1), int(end_y2)), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(frame, str(int(scores[i] * 100)) + "%", (int(end_x1), int(end_y1)), cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 255, 0), 2)
        return frame
    
    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 비디오 프레임을 읽을 수 없는 경우 루프 종료
            cap_h, cap_w, _ = frame.shape
            start_time = time()
            results = self.get_model_results(frame)
            frame = self.draw_rect(results, frame)
            frame = self.calc_distances(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            print('{0}x{1}, {2}, Inference : {3}ms, Distance : {4}m'.format(cap_w, cap_h, cls_name, round(fps/1000 * 100, 3), round(distance, 2)))
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Distance Estimation', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()  # 비디오 캡처 객체 해제
        cv2.destroyAllWindows()

# DistanceEstimationDetector 객체 생성
detector = DistanceEstimationDetector(video_path='C:/sinheechan.github.io-master/Car-Distance-Estimation_lee/input/people_street.mp4', model_path='yolov5s.pt')
detector()