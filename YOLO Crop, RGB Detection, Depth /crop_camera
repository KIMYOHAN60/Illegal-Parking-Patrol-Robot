import rclpy  # ROS2 Python í´ë¼ì´ì–¸íŠ¸ ë¼ì´ë¸ŒëŸ¬ë¦¬
from rclpy.node import Node  # ROS2 ë…¸ë“œ ë² ì´ìŠ¤ í´ëž˜ìŠ¤
from rclpy.executors import MultiThreadedExecutor  # ë©€í‹°ìŠ¤ë ˆë“œ ì‹¤í–‰ê¸°
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo  # ROS2 ì´ë¯¸ì§€ ë©”ì‹œì§€ íƒ€ìž…
from std_msgs.msg import String
from cv_bridge import CvBridge  # ROS ì´ë¯¸ì§€ :ì–‘ë°©í–¥_í™”ì‚´í‘œ: OpenCV ë³€í™˜ ë¸Œë¦¬ì§€
import cv2  # OpenCV ë¼ì´ë¸ŒëŸ¬ë¦¬
import numpy as np  # ìˆ˜ì¹˜ ì—°ì‚° ë¼ì´ë¸ŒëŸ¬ë¦¬
import math  # ìˆ˜í•™ í•¨ìˆ˜
import os  # ìš´ì˜ì²´ì œ ì¸í„°íŽ˜ì´ìŠ¤
import sys  # ì‹œìŠ¤í…œ ìƒí˜¸ìž‘ìš© 
import threading  # íŒŒì´ì¬ ìŠ¤ë ˆë”©
from ultralytics import YOLO  # YOLO ê°ì²´ ê°ì§€ ëª¨ë¸
# ================================
# ì„¤ì • ìƒìˆ˜
# ================================
MODEL_PATH = '/home/moonseungyeon/Downloads/11n_5_19.pt'      # YOLO ëª¨ë¸ íŒŒì¼ ê²½ë¡œ
RGB_TOPIC = '/robot3/oakd/rgb/preview/image_raw'       # RGB ì´ë¯¸ì§€ í† í”½
DEPTH_TOPIC = '/robot3/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot3/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0                                    # ê´€ì‹¬ ê°ì²´ í´ëž˜ìŠ¤ ID (0=ìžë™ì°¨)
NORMALIZE_DEPTH_RANGE = 3.0                            # ê¹Šì´ ì •ê·œí™” ë²”ìœ„ (m)
INTRUSION_THRESHOLD = 0.10                             # ì¹¨ë²” íŒë‹¨ ìž„ê³„ì¹˜ (10%)
BOX_PLUS = 25
TARGET_CLASS_ID = 0
NORMALIZE_DEPTH_RANGE = 3.0  # meters

class YoloDepthGreenDetector(Node):
    def __init__(self):
        super().__init__('yolo_depth_green_detector')
        # ëª¨ë¸ íŒŒì¼ í™•ì¸
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"ëª¨ë¸ì„ ì°¾ì„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤: {MODEL_PATH}")
            sys.exit(1)
        # YOLO ë¡œë“œ
        self.model = YOLO(MODEL_PATH)
        self.class_names = getattr(self.model, 'names', [])
        # CvBridge ì´ˆê¸°í™”
        self.bridge = CvBridge()
        # í† í”½ êµ¬ë…
        self.rgb_sub = self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, qos_profile_sensor_data)
        self.publisher_ = self.create_publisher(String, 'pose_label', 10)
        # ì´ë¯¸ì§€ ë²„í¼ ë° ë½
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
        self.get_logger().info(f":ë Œì¹˜: CameraInfo ìˆ˜ì‹ ë¨: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")


    def depth_callback(self, msg):
        """Depth ì´ë¯¸ì§€ ì½œë°±"""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f"Depth ë³€í™˜ ì˜¤ë¥˜: {e}")


    def rgb_callback(self, msg):
        """RGB ì´ë¯¸ì§€ ì½œë°±"""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                # depth ë³µì‚¬
                self.latest_depth = self.latest_depth.copy() if self.latest_depth is not None else None
            self.latest_rgb = img
        except Exception as e:
            self.get_logger().warn(f"RGB ë³€í™˜ ì˜¤ë¥˜: {e}")
    
    '''
    preview ì´ë¯¸ì§€ ì›ë³¸(í˜„ìž¬ëŠ” 360x360)ì„ ì½ì–´ ì˜¤ê³ ,x,yìµœëŒ€ ì¢Œí‘œë¥¼ ì½ì–´ì˜´.
    x,yì˜ ê°’ì„ ìž˜ëª»ìž…ë ¥í•´ì£¼ë©´ ë™ìž‘ì•ˆí•¨.
    initì— ìžˆëŠ”self.crop_y_point ë³€ê²½í•˜ë©´ ë³€ê²½ë˜ëŠ”ë° ì§€ê¸ˆ 150ìœ¼ë¡œ ë˜ì„œ 210x360ì´ ë¨. 
    '''
    def crop_image_bottom(self,img,a,b):
        # yì  ì•„ëž˜ ë¶€ë¶„ë§Œ ìžë¥´ê¸°
        cropped_img = img[self.crop_y_point:b, 0:b]
        return cropped_img
    
    '''
    ê°ì²´ì˜ ì¢Œí‘œ(x,y)ë¥¼ ë„£ì–´ì£¼ë©´ preview ì´ë¯¸ì§€ (a,b)í¬ê¸°, depth ì´ë¯¸ì§€ (c,d)í¬ê¸°
    coor_ratioëŠ” ë°°ìœ¨ì„ êµ¬í•˜ëŠ”ê±°ìž„. ê¸°ë³¸ì ìœ¼ë¡œ ê°™ì€ ì´ë¯¸ì§€ë¥¼ ë°›ëŠ” ì‚¬ì´ì¦ˆë¡œ ë³€ê²½í•˜ë©´ ë†’ì´ë¥¼ ê°™ê²Œ ë§žì¶œìˆ˜ ìžˆëŠ”ë° ì•ˆë§žìœ¼ë©´ ë°°ìœ¨ì„ ë‹¤ë¥´ê²Œ í•˜ë©´ë¨. d/b
    ì¤‘ì‹¬ì  ê¸°ì¤€ìœ¼ë¡œ ë‘ê°œì˜ ì¢Œí‘œê°€ ê°™ìŒ. ê·¸ëž˜ì„œ ì–‘ìª½ ì˜†ìœ¼ë¡œ ê°™ì€ë° aì— ë°°ìœ¨ì„ ê³±í•´ì„œ cì—ë¹¼ë©´ ì–‘ì˜†ì— ë‚ ë¼ê°„ ì´ë¯¸ì§€ ì¢Œí‘œë“¤ì˜ ê¸¸ì´ì¸ë° ì–‘ìª½ì´ë‹ˆ ë°˜ì„ ë‚˜ëˆ”. (c-coor_ratio*a)/2
    ìš°ì„  x,y ë‘˜ë‹¤ ë°°ìœ¨ì„ ê³±í•˜ê³  ê±°ê¸°ì— xëŠ” ì–‘ì˜† ë‚ ë¼ê°„ ì´ë¯¸ì§€ ì¢Œí‘œì¤‘ ë°˜ì„ ë”í•´ì£¼ê³ , yëŠ” í¬ë¡­ìœ¼ë¡œ ìƒìœ„ë¥¼ ë‚ ë¦¬ê³  ìœ„ì•„ëž˜ë¡œ ì´ë¯¸ì§€ë¥¼ ë”í•´ì¤¬ìœ¼ë‹ˆ í¬ë¡­ê¸¸ì´ì˜ ì ˆë°˜ì„ ë”í•´ì£¼ë©´ ì›ëž˜ ì¢Œí‘œì™€ ê°™ì•„ì§.
    '''
    def coor_correction(self,x,y,a,b,c,d):
        coor_ratio=d/b
        coor_add=(c-coor_ratio*a)/2
        return int(coor_ratio*x+coor_add),int(coor_ratio*y+(self.crop_y_point/2))
    
    '''
    
    '''
    def make_square_image(self, img):
        # ìœ„ì•„ëž˜ ë¹ˆ ê³µê°„ ì¶”ê°€ (ì´ ë†’ì´ 360pxë¡œ ë§žì¶¤) ì¤‘ì•™ ì¢Œí‘œê°€ ê°™ì€ ì´ë¯¸ì§€ê°€ ë˜ê²Œ í•˜ê¸°ìœ„í•´ì„œ ë§žì¶”ëŠ”ê±°ìž„.
        top_padding = (self.crop_y_point) // 2  # ìœ„ìª½ íŒ¨ë”©
        bottom_padding = (self.crop_y_point) // 2  # ì•„ëž˜ìª½ íŒ¨ë”©
        top_pad = np.ones((top_padding, img.shape[1], 3), dtype=np.uint8)* 255  # ê²€ì •ìƒ‰
        bottom_pad = np.ones((bottom_padding, img.shape[1], 3), dtype=np.uint8)* 255  # ê²€ì •ìƒ‰
        # ì„¸ë¡œë¡œ ìŒ“ì•„ì„œ 360x360 ì´ë¯¸ì§€ ìƒì„±
        result = np.vstack((top_pad, img, bottom_pad))
        return result
    
    
    def process_and_show(self):
        """ë„ë¡œ ë§ˆìŠ¤í¬ + YOLO ê°ì²´ ê²€ì¶œ + ì¹¨ë²” íŒë‹¨ + Depth ì‹œê°í™”"""
        with self.lock:
            rgb_img = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth_img = self.latest_depth.copy() if self.latest_depth is not None else None
        if (rgb_img is None)|(depth_img is None):
            return
        #rgb_img=cv2.resize(rgb_img,(720,720))
        # :ì¼: ì´ˆë¡ìƒ‰ ë„ë¡œ ì˜ì—­ ë§ˆìŠ¤í¬ ìƒì„± (HSV)

        #ì´ë¯¸ì§€ì˜ í¬ê¸°ë¥¼ ë°›ì•„ì˜¤ëŠ”ë° a,bê°€ x,yë¼ì„œ ì¢€ë” ì‰½ê²Œ ì´í•´í•˜ë ¤ê³  ë°›ì„ë•Œ b,a ë¡œí•´ì„œ ë°›ìŒ. c,dë„ ë™ì¼
        b,a=rgb_img.shape[:2]
        d,c=depth_img.shape[:2]
        # ìœ„ì— ì•ˆì“°ëŠ” ì´ë¯¸ì§€ í¬ë¡­í•´ì„œ ì§€ìš°ê¸°
        rgb_img = self.crop_image_bottom(rgb_img,a,b)
        # í¬ë¡­ëœ ì´ë¯¸ì§€ 360x360ìœ¼ë¡œ ë³€ê²½í•˜ê³  ì´ë¯¸ì§€ ìœ„ì•„ëž˜ë¡œ ì €ìž¥í•˜ê¸°.
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
        # :ë‘˜: YOLO ê²€ì¶œ
        results = self.model(rgb_img, stream=True, conf=0.7)
        object_count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != TARGET_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                #ì¤‘ì‹¬ ì¢Œí‘œë¥¼ ì•Œì•„ì„œ ë³€ê²½í•˜ê²Œ í•´ì£¼ëŠ”ê±°ë‹ˆê¹Œ ìŒ..... ì—¬ê¸°ë³´ë‹¨ ì½”ë“œì—ê°€ì„œ ì„¤ëª…ë³´ëŠ”ê²Œ ì´í•´ ë êº¼... ë„£ëŠ”ê±´ x,y ë°•ìŠ¤ ì¢Œí‘œë“¤ì„ í•©ì³ì„œ ì ˆë°˜ìœ¼ë¡œ ë‚˜ëˆ ì„œ ì¤‘ì•™ ì¢Œí‘œë¥¼ ì°¾ëŠ”ê±°ìž„.
                cx, cy = self.coor_correction((x1 + x2) // 2, (y1 + y2) // 2,a,b,c,d)
                #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                conf = math.ceil(box.conf[0] * 100) / 100
                # :ì…‹: ë„ë¡œ ì˜ì—­ ì¹¨ë²” íŒë‹¨ (ë°•ìŠ¤ ë‚´ ì´ˆë¡ìƒ‰ ë§ˆìŠ¤í¬ ë¹„ìœ¨) ê°ì²´ ë°•ìŠ¤ ë‚´ë¶€ê°€ ì–¼ë§ˆë‚˜ ì´ˆë¡ìƒ‰ì„ í¬í•¨í•˜ëƒ ì¸ë°
                # BOX_PLUSë¥¼ ì‚¬ìš©í•´ì„œ ì•„ëž˜ëž‘ ì–‘ì˜†ì„ ëŠ˜ë ¤ì„œíŒë‹¨í•¨ ê·¸ë¦¬ê³  ìœ„ëŠ” ëœë³¼ë ¤ê³  ì•„ëž˜ê¸°ì¤€ 60% ë†’ì´ê¹Œì§€ë§Œ ë´„.
                roi_mask = green_mask[y1+int((y2-y1)*6/10):y2+BOX_PLUS, x1-BOX_PLUS:x2+BOX_PLUS]
                # ì•„ëž˜ëŠ” ëŒ€ê°• 0ì¡´ ë§ˆìŠ¤í¬í•œê±¸ ë³´ê³  ì´ í”½ìƒ ìˆ˜ì— ì´ê²Œ ì „ì²´ ì—ì„œ ë¹„ìœ¨ì´ ì–¼ë§ˆë‚˜ ë˜ë‚˜ í•´ì„œ (INTRUSION_THRESHOLD)10%í¬í•¨í•˜ë©´ ë§žë‹¤ë¡œ ì¸ì‹í•¨.
                pixel_count = cv2.countNonZero(roi_mask)
                total_pixels = (y2 - y1) * (x2 - x1)
                ratio = pixel_count / total_pixels if total_pixels > 0 else 0
                on_path = (ratio > INTRUSION_THRESHOLD)
                # :ë„·: ê¹Šì´ ê³„ì‚° (ì¤‘ì•™ ROI í‰ê· )
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
                # :ë‹¤ì„¯: 3D ì¢Œí‘œ ë³€í™˜ ë° ë¡œê·¸
                if depth_val is not None:
                    z = depth_val
                    x = (cx - self.cx) * z / self.fx
                    y = (cy - self.cy) * z / self.fy
                    self.publish_pose_label(x,y,z,label)
                    self.get_logger().info(f"[TF] {label}: x={x:.2f}, y={y:.2f}, z={z:.2f}, on_path={on_path}")
                # :ì—¬ì„¯: ì¹¨ë²” ì‹œ ê²½ê³  ë¡œê·¸
                if on_path:
                    self.get_logger().warn(":ê²½ê´‘ë“±: ë¶ˆë²• ì°¨ëŸ‰ í™•ì¸!")
                # :ì¼ê³±: ì‹œê°í™”: ë°•ìŠ¤, ë¼ë²¨, ì¹¨ë²” ì—¬ë¶€ í‘œì‹œ
                box_color = (0, 0, 255) if on_path else (255, 255, 255)
                text = f"{label} {conf:.2f}" + (", illegal" if on_path else "")
                if depth_val is not None:
                    text += f" {depth_val:.2f}m"
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(rgb_img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                object_count += 1
        # ê°ì²´ ê°œìˆ˜ í‘œì‹œ
        cv2.putText(rgb_img, f"Objects: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # :ì—¬ëŸ: ê¹Šì´ ì˜ìƒ ì»¬ëŸ¬ë§µ ì‹œê°í™”
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
    node = YoloDepthGreenDetector()  # ë˜ëŠ” YoloDepthViewer ì¤‘ í•˜ë‚˜ë§Œ ì‚¬ìš©
    ros_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    ros_thread.start()
    try:
        while rclpy.ok() and not node.should_shutdown:
            node.process_and_show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                node.should_shutdown = True
                node.get_logger().info("Q ëˆŒëŸ¬ ì¢…ë£Œí•©ë‹ˆë‹¤.")
                break
    except KeyboardInterrupt:
        node.get_logger().info("í‚¤ë³´ë“œ ì¸í„°ëŸ½íŠ¸ë¡œ ì¢…ë£Œí•©ë‹ˆë‹¤.")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
        ros_thread.join()
if __name__ == '__main__':
    main()
