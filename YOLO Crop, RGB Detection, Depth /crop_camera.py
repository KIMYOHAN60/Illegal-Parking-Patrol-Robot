import rclpy  # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node  # ROS2 노드 베이스 클래스
from rclpy.executors import MultiThreadedExecutor  # 멀티스레드 실행기
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo  # ROS2 이미지 메시지 타입
from std_msgs.msg import String
from cv_bridge import CvBridge  # ROS 이미지 :양방향_화살표: OpenCV 변환 브리지
import cv2  # OpenCV 라이브러리
import numpy as np  # 수치 연산 라이브러리
import math  # 수학 함수
import os  # 운영체제 인터페이스
import sys  # 시스템 상호작용 
import threading  # 파이썬 스레딩
from ultralytics import YOLO  # YOLO 객체 감지 모델
# ================================
# 설정 상수
# ================================
MODEL_PATH = '/home/moonseungyeon/Downloads/11n_5_19.pt'      # YOLO 모델 파일 경로
RGB_TOPIC = '/robot3/oakd/rgb/preview/image_raw'       # RGB 이미지 토픽
DEPTH_TOPIC = '/robot3/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot3/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0                                    # 관심 객체 클래스 ID (0=자동차)
NORMALIZE_DEPTH_RANGE = 3.0                            # 깊이 정규화 범위 (m)
INTRUSION_THRESHOLD = 0.10                             # 침범 판단 임계치 (10%)
BOX_PLUS = 25
TARGET_CLASS_ID = 0
NORMALIZE_DEPTH_RANGE = 3.0  # meters

class YoloDepthGreenDetector(Node):
    def __init__(self):
        super().__init__('yolo_depth_green_detector')
        # 모델 파일 확인
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"모델을 찾을 수 없습니다: {MODEL_PATH}")
            sys.exit(1)
        # YOLO 로드
        self.model = YOLO(MODEL_PATH)
        self.class_names = getattr(self.model, 'names', [])
        # CvBridge 초기화
        self.bridge = CvBridge()
        # 토픽 구독
        self.rgb_sub = self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(String, 'pose_label', 10)
        # 이미지 버퍼 및 락
        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()
        self.should_shutdown = False
        self.crop_y_point = 150


    def publish_pose_label(self,x,y,z,label):
        msg = String()
        msg.data = f"{x},{y},{z},{label}"
        self.publisher_.publish(msg)

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(f":렌치: CameraInfo 수신됨: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")


    def depth_callback(self, msg):
        """Depth 이미지 콜백"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f"Depth 변환 오류: {e}")


    def rgb_callback(self, msg):
        """RGB 이미지 콜백"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                # depth 복사
                self.latest_depth = self.latest_depth.copy() if self.latest_depth is not None else None
            self.latest_rgb = img
        except Exception as e:
            self.get_logger().warn(f"RGB 변환 오류: {e}")
    
    '''
    preview 이미지 원본(현재는 360x360)을 읽어 오고,x,y최대 좌표를 읽어옴.
    x,y의 값을 잘못입력해주면 동작안함.
    init에 있는self.crop_y_point 변경하면 변경되는데 지금 150으로 되서 210x360이 됨. 
    '''
    def crop_image_bottom(self,img,a,b):
        # y점 아래 부분만 자르기
        cropped_img = img[self.crop_y_point:b, 0:b]
        return cropped_img
    
    '''
    객체의 좌표(x,y)를 넣어주면 preview 이미지 (a,b)크기, depth 이미지 (c,d)크기
    coor_ratio는 배율을 구하는거임. 기본적으로 같은 이미지를 받는 사이즈로 변경하면 높이를 같게 맞출수 있는데 안맞으면 배율을 다르게 하면됨. d/b
    중심점 기준으로 두개의 좌표가 같음. 그래서 양쪽 옆으로 같은데 a에 배율을 곱해서 c에빼면 양옆에 날라간 이미지 좌표들의 길이인데 양쪽이니 반을 나눔. (c-coor_ratio*a)/2
    우선 x,y 둘다 배율을 곱하고 거기에 x는 양옆 날라간 이미지 좌표중 반을 더해주고, y는 크롭으로 상위를 날리고 위아래로 이미지를 더해줬으니 크롭길이의 절반을 더해주면 원래 좌표와 같아짐.
    '''
    def coor_correction(self,x,y,a,b,c,d):
        coor_ratio=d/b
        coor_add=(c-coor_ratio*a)/2
        return int(coor_ratio*x+coor_add),int(coor_ratio*y+(self.crop_y_point/2))
    
    '''
    
    '''
    def make_square_image(self, img):
        # 위아래 빈 공간 추가 (총 높이 360px로 맞춤) 중앙 좌표가 같은 이미지가 되게 하기위해서 맞추는거임.
        top_padding = (self.crop_y_point) // 2  # 위쪽 패딩
        bottom_padding = (self.crop_y_point) // 2  # 아래쪽 패딩
        top_pad = np.ones((top_padding, img.shape[1], 3), dtype=np.uint8)* 255  # 검정색
        bottom_pad = np.ones((bottom_padding, img.shape[1], 3), dtype=np.uint8)* 255  # 검정색
        # 세로로 쌓아서 360x360 이미지 생성
        result = np.vstack((top_pad, img, bottom_pad))
        return result
    
    
    def process_and_show(self):
        """도로 마스크 + YOLO 객체 검출 + 침범 판단 + Depth 시각화"""
        with self.lock:
            rgb_img = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth_img = self.latest_depth.copy() if self.latest_depth is not None else None
        if (rgb_img is None)|(depth_img is None):
            return
        #rgb_img=cv2.resize(rgb_img,(720,720))
        # :일: 초록색 도로 영역 마스크 생성 (HSV)

        #이미지의 크기를 받아오는데 a,b가 x,y라서 좀더 쉽게 이해하려고 받을때 b,a 로해서 받음. c,d도 동일
        b,a=rgb_img.shape[:2]
        d,c=depth_img.shape[:2]
        # 위에 안쓰는 이미지 크롭해서 지우기
        rgb_img = self.crop_image_bottom(rgb_img,a,b)
        # 크롭된 이미지 360x360으로 변경하고 이미지 위아래로 저장하기.
        rgb_img = self.make_square_image(rgb_img)

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 64])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        overlay = rgb_img.copy()
        overlay[green_mask>0] = (0,0,255)
        cv2.addWeighted(overlay, 0.3, rgb_img, 0.7, 0, rgb_img)
        # :둘: YOLO 검출
        results = self.model(rgb_img, stream=True, conf=0.7)
        object_count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != TARGET_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                #중심 좌표를 알아서 변경하게 해주는거니까 음..... 여기보단 코드에가서 설명보는게 이해 될꺼... 넣는건 x,y 박스 좌표들을 합쳐서 절반으로 나눠서 중앙 좌표를 찾는거임.
                cx, cy = self.coor_correction((x1 + x2) // 2, (y1 + y2) // 2,a,b,c,d)
                #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                conf = math.ceil(box.conf[0] * 100) / 100
                # :셋: 도로 영역 침범 판단 (박스 내 초록색 마스크 비율) 객체 박스 내부가 얼마나 초록색을 포함하냐 인데
                # BOX_PLUS를 사용해서 아래랑 양옆을 늘려서판단함 그리고 위는 덜볼려고 아래기준 60% 높이까지만 봄.
                roi_mask = green_mask[y1+int((y2-y1)*6/10):y2+BOX_PLUS, x1-BOX_PLUS:x2+BOX_PLUS]
                # 아래는 대강 0존 마스크한걸 보고 총 픽샐 수에 이게 전체 에서 비율이 얼마나 되나 해서 (INTRUSION_THRESHOLD)10%포함하면 맞다로 인식함.
                pixel_count = cv2.countNonZero(roi_mask)
                total_pixels = (y2 - y1) * (x2 - x1)
                ratio = pixel_count / total_pixels if total_pixels > 0 else 0
                on_path = (ratio > INTRUSION_THRESHOLD)
                # :넷: 깊이 계산 (중앙 ROI 평균)
                depth_val = None
                if depth_img is not None:
                    roi_size = 5
                    x_start = max(cx - roi_size , 0)
                    x_end = min(cx + roi_size, depth_img.shape[1])
                    y_start = max(cy - roi_size, 0)
                    y_end = min(cy + roi_size, depth_img.shape[0])
                    depth_roi = depth_img[y_start:y_end, x_start:x_end]
                    valid = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
                    if valid.size > 0:
                        depth_val = np.mean(valid) / 1000.0  # mm -> m
                # :다섯: 3D 좌표 변환 및 로그
                if depth_val is not None:
                    z = depth_val
                    x = (cx - self.cx) * z / self.fx
                    y = (cy - self.cy) * z / self.fy
                    self.publish_pose_label(x,y,z,label)
                    self.get_logger().info(f"[TF] {label}: x={x:.2f}, y={y:.2f}, z={z:.2f}, on_path={on_path}")
                # :여섯: 침범 시 경고 로그
                if on_path:
                    self.get_logger().warn(":경광등: 불법 차량 확인!")
                # :일곱: 시각화: 박스, 라벨, 침범 여부 표시
                box_color = (0, 0, 255) if on_path else (255, 255, 255)
                text = f"{label} {conf:.2f}" + (", illegal" if on_path else "")
                if depth_val is not None:
                    text += f" {depth_val:.2f}m"
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(rgb_img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                object_count += 1
        # 객체 개수 표시
        cv2.putText(rgb_img, f"Objects: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # :여덟: 깊이 영상 컬러맵 시각화
        if depth_img is not None:
            vis_depth = np.nan_to_num(depth_img, nan=0.0)
            vis_depth[vis_depth < 300] = 0
            vis_depth = np.clip(vis_depth, 0, NORMALIZE_DEPTH_RANGE * 1000)
            vis_depth_norm = (vis_depth / (NORMALIZE_DEPTH_RANGE * 1000) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(vis_depth_norm, cv2.COLORMAP_JET)
            combined = np.hstack((rgb_img, depth_colored))
            cv2.imshow("YOLO+Depth+GreenMask", combined)
        else:
            cv2.imshow("YOLO+Depth+GreenMask", rgb_img)


def ros_spin_thread(node):
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()


def main():
    rclpy.init()
    node = YoloDepthGreenDetector()  # 또는 YoloDepthViewer 중 하나만 사용
    ros_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    ros_thread.start()
    try:
        while rclpy.ok() and not node.should_shutdown:
            node.process_and_show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                node.should_shutdown = True
                node.get_logger().info("Q 눌러 종료합니다.")
                break
    except KeyboardInterrupt:
        node.get_logger().info("키보드 인터럽트로 종료합니다.")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
        ros_thread.join()
if __name__ == '__main__':
    main()
