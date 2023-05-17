# -*- coding: UTF-8 -*-
"""
  @Author: hawcat
  @Date  : 2023/3/16 20:43
  @version V1.1
"""

# 公共包
import random
import sys
import threading
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# YoloV5系列包
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
# OpenPose系列包
from pose_visualizer import TfPoseVisualizer
from pathlib import Path
import argparse

if True:  # Include project path
    import os

    ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *  # Import all sklearn related libraries


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


file_path = Path.cwd()
from Pose.utils import load_pretrain_model

# YOlO模型地址
model_path = '../weights/best_small.pt'

# OpenPose按钮事件
clicked = 0
cam_parse = 'video'
video_path = '../data_openpose/test.mp4'


# 添加一个关于界面
# 窗口主类
class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('S.M.A.R.T 智能课堂学生行为识别')
        self.resize(1280, 900)
        self.setWindowIcon(QIcon("../UI/smart.png"))
        # 图片读取进程
        self.output_size = 600
        self.img2predict = ""
        # 空字符串会自己进行选择，首选cuda
        self.device = ''
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        # 检测视频的线程
        self.threading = None
        # 是否跳出当前循环的线程
        self.jump_threading: bool = False

        self.image_size = 800
        self.confidence = 0.25
        self.iou_threshold = 0.45
        # 指明模型加载的位置的设备
        self.model = self.model_load(weights=model_path,
                                     device=self.device)
        self.initUI()
        self.reset_vid()

    @torch.no_grad()
    def model_load(self,
                   weights="",  # model.pt path(s)
                   device='0',  # yolo加载cuda device, i.e. 0 or 0,1,2,3 or cpu
                   ):
        """
		模型初始化
		"""
        device = self.device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.image_size = check_img_size(self.image_size, s=self.stride)  # check img_size
        if half:
            model.half()  # to FP16
        # Run inference
        if device.type != 'cpu':
            print("Run inference")
            model(torch.zeros(1, 3, self.image_size, self.image_size).to(device).type_as(
                next(model.parameters())))  # run once
        print("模型加载完成!")
        return model

    def reset_vid(self):
        """
		界面重置事件
		"""
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
        self.vid_source = '0'
        self.disable_btn(self.det_img_button)
        self.disable_btn(self.vid_start_stop_btn)
        self.jump_threading = False

    def initUI(self):
        """
		界面初始化
		"""
        # 图片检测子界面
        font_title = QFont('微软雅黑', 16)
        font_main = QFont('微软雅黑', 14)
        font_general = QFont('微软雅黑', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("Yolo-手机图片检测")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("./UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        self.left_img.setMinimumSize(600, 600)
        self.left_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
        mid_img_layout.addWidget(self.left_img)
        self.right_img.setMinimumSize(600, 600)
        self.right_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        self.up_img_button = QPushButton("上传图片")
        self.det_img_button = QPushButton("开始检测")
        self.up_img_button.clicked.connect(self.upload_img)
        self.det_img_button.clicked.connect(self.detect_img)
        self.up_img_button.setFont(font_main)
        self.det_img_button.setFont(font_main)
        self.up_img_button.setStyleSheet("QPushButton{color:white}"
                                         "QPushButton:hover{background-color: rgb(2,110,180);}"
                                         "QPushButton{background-color:rgb(48,124,208)}"
                                         "QPushButton{border:2px}"
                                         "QPushButton{border-radius:5px}"
                                         "QPushButton{padding:5px 5px}"
                                         "QPushButton{margin:5px 5px}")
        self.det_img_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.up_img_button)
        img_detection_layout.addWidget(self.det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 视频识别界面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("Yolo-手机视频流检测")
        vid_title.setFont(font_title)
        self.left_vid_img = QLabel()
        self.right_vid_img = QLabel()
        self.left_vid_img.setPixmap(QPixmap("./UI/up.jpeg"))
        self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
        self.left_vid_img.setAlignment(Qt.AlignCenter)
        self.left_vid_img.setMinimumSize(600, 600)
        self.left_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
        self.right_vid_img.setAlignment(Qt.AlignCenter)
        self.right_vid_img.setMinimumSize(600, 600)
        self.right_vid_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        mid_img_layout.addWidget(self.left_vid_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_vid_img)
        mid_img_widget.setLayout(mid_img_layout)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_start_stop_btn = QPushButton("启动/停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_start_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_start_stop_btn.setStyleSheet("QPushButton{color:white}"
                                              "QPushButton:hover{background-color: rgb(2,110,180);}"
                                              "QPushButton{background-color:rgb(48,124,208)}"
                                              "QPushButton{border:2px}"
                                              "QPushButton{border-radius:5px}"
                                              "QPushButton{padding:5px 5px}"
                                              "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_start_stop_btn.clicked.connect(self.start_or_stop)

        # 添加fps显示
        fps_container = QWidget()
        fps_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
        fps_container_layout = QHBoxLayout()
        fps_container.setLayout(fps_container_layout)
        # 左容器
        fps_left_container = QWidget()
        fps_left_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
        fps_left_container_layout = QHBoxLayout()
        fps_left_container.setLayout(fps_left_container_layout)

        # 右容器
        fps_right_container = QWidget()
        fps_right_container.setStyleSheet("QWidget{background-color: #f6f8fa;}")
        fps_right_container_layout = QHBoxLayout()
        fps_right_container.setLayout(fps_right_container_layout)

        # 将左容器和右容器添加到fps_container_layout中
        fps_container_layout.addWidget(fps_left_container)
        fps_container_layout.addStretch(0)
        fps_container_layout.addWidget(fps_right_container)

        # 左容器中添加fps显示
        raw_fps_label = QLabel("原始帧率:")
        raw_fps_label.setFont(font_general)
        raw_fps_label.setAlignment(Qt.AlignLeft)
        raw_fps_label.setStyleSheet("QLabel{margin-left:80px}")
        self.raw_fps_value = QLabel("0")
        self.raw_fps_value.setFont(font_general)
        self.raw_fps_value.setAlignment(Qt.AlignLeft)
        fps_left_container_layout.addWidget(raw_fps_label)
        fps_left_container_layout.addWidget(self.raw_fps_value)

        # 右容器中添加fps显示
        detect_fps_label = QLabel("检测帧率:")
        detect_fps_label.setFont(font_general)
        detect_fps_label.setAlignment(Qt.AlignRight)
        self.detect_fps_value = QLabel("0")
        self.detect_fps_value.setFont(font_general)
        self.detect_fps_value.setAlignment(Qt.AlignRight)
        self.detect_fps_value.setStyleSheet("QLabel{margin-right: 48px}")
        fps_right_container_layout.addWidget(detect_fps_label)
        fps_right_container_layout.addWidget(self.detect_fps_value)

        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(fps_container)
        vid_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_start_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # OpenPose界面
        openpose_widget = QWidget()
        openpose_layout = QVBoxLayout()
        openpose_title = QLabel("OpenPose-学生姿态检测")
        openpose_title.setFont(font_title)
        openpose_title.setFont(font_title)
        self.openpose_img = QLabel()
        self.openpose_img.setAlignment(Qt.AlignCenter)
        self.openpose_img.setMinimumSize(1200, 600)
        self.openpose_img.setStyleSheet("QLabel{background-color: #f6f8fa;}")
        self.camera_detection_btn = QPushButton("开/关摄像头")
        self.openpose_choose_btn = QPushButton("选择待检测数据")
        #self.pl_btn = QPushButton("批量检测")
        self.openpose_start_stop_btn = QPushButton("启动/停止检测")
        self.camera_detection_btn.setFont(font_main)
        self.openpose_choose_btn.setFont(font_main)
        self.openpose_start_stop_btn.setFont(font_main)
        self.camera_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.openpose_choose_btn.setStyleSheet("QPushButton{color:white}"
                                               "QPushButton:hover{background-color: rgb(2,110,180);}"
                                               "QPushButton{background-color:rgb(48,124,208)}"
                                               "QPushButton{border:2px}"
                                               "QPushButton{border-radius:5px}"
                                               "QPushButton{padding:5px 5px}"
                                               "QPushButton{margin:5px 5px}")
        # #self.pl_btn.setStyleSheet("QPushButton{color:white}"
        #                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
        #                                        "QPushButton{background-color:rgb(48,124,208)}"
        #                                        "QPushButton{border:2px}"
        #                                        "QPushButton{border-radius:5px}"
        #                                        "QPushButton{padding:5px 5px}"
        #                                        "QPushButton{margin:5px 5px}")
        self.openpose_start_stop_btn.setStyleSheet("QPushButton{color:white}"
                                                   "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                   "QPushButton{background-color:rgb(48,124,208)}"
                                                   "QPushButton{border:2px}"
                                                   "QPushButton{border-radius:5px}"
                                                   "QPushButton{padding:5px 5px}"
                                                   "QPushButton{margin:5px 5px}")
        self.camera_detection_btn.clicked.connect(self.set_cam)
        #self.pl_btn.clicked.connect(self.set_vio)
        self.openpose_choose_btn.clicked.connect(self.choose_video)
        self.openpose_start_stop_btn.clicked.connect(self.startopenpose)
        openpose_layout.addWidget(self.openpose_img)
        openpose_layout.addWidget(openpose_title, alignment=Qt.AlignCenter)
        openpose_layout.addWidget(self.camera_detection_btn)
        openpose_layout.addWidget(self.openpose_choose_btn)
#        openpose_layout.addWidget(self.pl_btn)
        openpose_layout.addWidget(self.openpose_start_stop_btn)
        openpose_widget.setLayout(openpose_layout)

        # 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(
            '工作鸣谢\n\n 欢迎使用该系统 \n\n 该系统是基于YoloV5和OpenPose的智能课堂学生行为识别 \n\n可目标检测使用手机的场景、识别学生课堂常见行为')  # 修改欢迎词语
        about_title.setFont(QFont('微软雅黑', 18))
        about_title.setAlignment(Qt.AlignCenter)
        # yolov5
        label_thanks1 = QLabel()
        label_thanks1.setAlignment(Qt.AlignCenter)
        label_thanks1.setText("<a href='https://github.com/ultralytics/yolov5'> ultralytics/yolov5(official)</a>")
        label_thanks1.setFont(QFont('楷体', 16))
        label_thanks1.setOpenExternalLinks(True)
        # OpenPose
        label_thanks2 = QLabel()
        label_thanks2.setAlignment(Qt.AlignCenter)
        label_thanks2.setText(
            "<a href='https://github.com/CMU-Perceptual-Computing-Lab/openpose'> CMU-Perceptual-Computing-Lab/openpose(official)</a>")
        label_thanks2.setFont(QFont('楷体', 16))
        label_thanks2.setOpenExternalLinks(True)
        # Tfopenposehttps://github.com/Zehaos/MobileNet
        label_thanks3 = QLabel()
        label_thanks3.setAlignment(Qt.AlignCenter)
        label_thanks3.setText(
            "<a href='https://github.com/Zumbalamambo/tf-openpose'> Zumbalamambo/tf-openpose(tensorflow版本的OpenPose实现)</a>")
        label_thanks3.setFont(QFont('楷体', 16))
        label_thanks3.setOpenExternalLinks(True)
        # Mobile-net https://github.com/Zehaos/MobileNet
        label_thanks4 = QLabel()
        label_thanks4.setAlignment(Qt.AlignCenter)
        label_thanks4.setText(
            "<a href='https://github.com/Zehaos/MobileNet'> Zehaos/MobileNet(区别于传统CNN的新型轻量级卷积网络)</a>")
        label_thanks4.setFont(QFont('楷体', 16))
        label_thanks4.setOpenExternalLinks(True)
        # Yolov3 https://github.com/coldlarry/YOLOv3-complete-pruning
        label_thanks5 = QLabel()
        label_thanks5.setAlignment(Qt.AlignCenter)
        label_thanks5.setText(
            "<a href='https://github.com/coldlarry/YOLOv3-complete-pruning'> coldlarry/YOLOv3-complete-pruning(剪枝参考)</a>")
        label_thanks5.setFont(QFont('楷体', 16))
        label_thanks5.setOpenExternalLinks(True)
        # UI https://github.com/mpj1234/yolov5-5.0-simpleUI
        label_thanks6 = QLabel()
        label_thanks6.setAlignment(Qt.AlignCenter)
        label_thanks6.setText(
            "<a href='https://github.com/mpj1234/yolov5-5.0-simpleUI'> mpj1234/yolov5-5.0-simpleUI(很棒的UI)</a>")
        label_thanks6.setFont(QFont('楷体', 16))
        label_thanks6.setOpenExternalLinks(True)
        about_layout.addWidget(label_thanks1)
        about_layout.addWidget(label_thanks2)
        about_layout.addWidget(label_thanks3)
        about_layout.addWidget(label_thanks4)
        about_layout.addWidget(label_thanks5)
        about_layout.addWidget(label_thanks6)

        label_super = QLabel()  # 更换作者信息
        label_super.setText("<a href='https://hawcat.cn'>我的博客</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addStretch()
        about_layout.addWidget(label_super)

        about_widget.setLayout(about_layout)

        self.addTab(img_detection_widget, 'Yolo-图片检测')
        self.addTab(vid_detection_widget, 'Yolo-视频流检测')
        self.addTab(openpose_widget, 'OpenPose行为识别')
        self.addTab(about_widget, '开源列表')
        self.setTabIcon(0, QIcon('../UI/pic.png'))
        self.setTabIcon(1, QIcon('../UI/vid.png'))
        self.setTabIcon(2, QIcon('../UI/openpose.png'))
        self.setTabIcon(3, QIcon('../UI/thanks.png'))

    def disable_btn(self, pushButton: QPushButton):
        pushButton.setDisabled(True)
        pushButton.setStyleSheet("QPushButton{background-color: rgb(2,110,180);}")

    def enable_btn(self, pushButton: QPushButton):
        pushButton.setEnabled(True)
        pushButton.setStyleSheet(
            "QPushButton{background-color: rgb(48,124,208);}"
            "QPushButton{color:white}"
        )

    def detect(self, source: str, left_img: QLabel, right_img: QLabel):
        """
		@param source: file/dir/URL/glob, 0 for webcam
		@param left_img: 将左侧QLabel对象传入，用于显示图片
		@param right_img: 将右侧QLabel对象传入，用于显示图片
		"""
        model = self.model
        img_size = [self.image_size, self.image_size]  # inference size (pixels)
        conf_threshold = self.confidence  # confidence threshold
        iou_threshold = self.iou_threshold  # NMS IOU threshold
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference

        half = device.type != 'cpu'  # half precision only supported on CUDA

        if source == "":
            self.disable_btn(self.det_img_button)
            QMessageBox.warning(self, "请上传", "请先上传视频或图片再进行检测")
        else:
            source = str(source)
            webcam = source.isnumeric()

            # Set Dataloader
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=img_size, stride=self.stride)
            else:
                dataset = LoadImages(source, img_size=img_size, stride=self.stride)
            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # 用来记录处理的图片数量
            count = 0
            # 计算帧率开始时间
            fps_start_time = time.time()
            for path, img, im0s, vid_cap in dataset:
                # 直接跳出for，结束线程
                if self.jump_threading:
                    # 清除状态
                    self.jump_threading = False
                    break
                count += 1
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=classes, agnostic=agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        s, im0 = '%g: ' % i, im0s[i].copy()
                    else:
                        s, im0 = '', im0s.copy()

                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    if webcam or vid_cap is not None:
                        if webcam:  # batch_size >= 1
                            img = im0s[i]
                        else:
                            img = im0s
                        img = self.resize_img(img)
                        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
                                     QImage.Format_RGB888)
                        left_img.setPixmap(QPixmap.fromImage(img))
                        # 计算一次帧率
                        if count % 10 == 0:
                            fps = int(10 / (time.time() - fps_start_time))
                            self.detect_fps_value.setText(str(fps))
                            fps_start_time = time.time()
                    # 归一化处理
                    img = self.resize_img(im0)
                    img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1],
                                 QImage.Format_RGB888)
                    right_img.setPixmap(QPixmap.fromImage(img))

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 使用完摄像头释放资源
            if webcam:
                for cap in dataset.caps:
                    cap.release()
            else:
                dataset.cap and dataset.cap.release()

    def resize_img(self, img):
        """
		调整图片大小，方便用来显示
		@param img: 需要调整的图片
		"""
        resize_scale = min(self.output_size / img.shape[0], self.output_size / img.shape[1])
        img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def upload_img(self):
        """
		上传图片
		"""
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            self.img2predict = fileName
            # 将上传照片和进行检测做成互斥的
            self.enable_btn(self.det_img_button)
            self.disable_btn(self.up_img_button)
            # 进行左侧原图展示
            img = cv2.imread(fileName)
            # 应该调整一下图片的大小
            img = self.resize_img(img)
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
            self.left_img.setPixmap(QPixmap.fromImage(img))
            # 上传图片之后右侧的图片重置
            self.right_img.setPixmap(QPixmap("./UI/right.jpeg"))

    def detect_img(self):
        """
		检测图片
		"""
        # 重置跳出线程状态，防止其他位置使用的影响
        self.jump_threading = False
        self.detect(self.img2predict, self.left_img, self.right_img)
        # 将上传照片和进行检测做成互斥的
        self.enable_btn(self.up_img_button)
        self.disable_btn(self.det_img_button)

    def open_mp4(self):
        """
		开启视频文件检测事件
		"""
        print("开启视频文件检测")
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.disable_btn(self.webcam_detection_btn)
            self.disable_btn(self.mp4_detection_btn)
            self.enable_btn(self.vid_start_stop_btn)
            # 生成读取视频对象
            cap = cv2.VideoCapture(fileName)
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 显示原始视频帧率
            self.raw_fps_value.setText(str(fps))
            if cap.isOpened():
                # 读取一帧用来提前左侧展示
                ret, raw_img = cap.read()
                cap.release()
            else:
                QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
                self.disable_btn(self.vid_start_stop_btn)
                self.enable_btn(self.webcam_detection_btn)
                self.enable_btn(self.mp4_detection_btn)
                return
            # 应该调整一下图片的大小
            img = self.resize_img(np.array(raw_img))
            img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
            self.left_vid_img.setPixmap(QPixmap.fromImage(img))
            # 上传图片之后右侧的图片重置
            self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))
            self.vid_source = fileName
            self.jump_threading = False

    def open_cam(self):
        """
		打开摄像头事件
		"""
        print("打开摄像头")
        self.disable_btn(self.webcam_detection_btn)
        self.disable_btn(self.mp4_detection_btn)
        self.enable_btn(self.vid_start_stop_btn)
        self.vid_source = "0"
        self.jump_threading = False
        # 生成读取视频对象
        cap = cv2.VideoCapture(0)
        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 显示原始视频帧率
        self.raw_fps_value.setText(str(fps))
        if cap.isOpened():
            # 读取一帧用来提前左侧展示
            ret, raw_img = cap.read()
            cap.release()
        else:
            QMessageBox.warning(self, "需要重新上传", "请重新选择视频文件")
            self.disable_btn(self.vid_start_stop_btn)
            self.enable_btn(self.webcam_detection_btn)
            self.enable_btn(self.mp4_detection_btn)
            return
        # 应该调整一下图片的大小
        img = self.resize_img(np.array(raw_img))
        img = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format_RGB888)
        self.left_vid_img.setPixmap(QPixmap.fromImage(img))
        # 上传图片之后右侧的图片重置
        self.right_vid_img.setPixmap(QPixmap("./UI/right.jpeg"))

    def start_or_stop(self):
        """
		启动或者停止事件
		"""
        print("启动或者停止")
        if self.threading is None:
            # 创造并启动一个检测视频线程
            self.jump_threading = False
            self.threading = threading.Thread(target=self.detect_vid)
            self.threading.start()
            self.disable_btn(self.webcam_detection_btn)
            self.disable_btn(self.mp4_detection_btn)
        else:
            # 停止当前线程
            # 线程属性置空，恢复状态
            self.threading = None
            self.jump_threading = True
            self.enable_btn(self.webcam_detection_btn)
            self.enable_btn(self.mp4_detection_btn)

    def detect_vid(self):
        """
		视频检测
		视频和摄像头的主函数是一样的，不过是传入的source不同罢了
		"""
        print("视频开始检测")
        self.detect(self.vid_source, self.left_vid_img, self.right_vid_img)
        print("视频检测结束")
        # 执行完进程，刷新一下和进程有关的状态，只有self.threading是None，
        # 才能说明是正常结束的线程，需要被刷新状态
        if self.threading is not None:
            self.start_or_stop()

    def closeEvent(self, event):
        """
		界面关闭事件
		"""
        reply = QMessageBox.question(
            self,
            '退出程序',
            "Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.jump_threading = True
            self.close()
            event.accept()
        else:
            event.ignore()

    def choose_video(self):
        global file_path
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.mkv *.avi')
        file_path = os.path.basename(fileName)

    def set_cam(self):
        global cam_parse
        cam_parse = 'webcam'



    def startopenpose(self):
        clicked = 1

        def get_command_line_arguments():
            # 参数
            def parse_args():
                parser = argparse.ArgumentParser(
                    description="Test action recognition on \n"
                                "(1) a video, (2) a folder of images, (3) or web camera.")
                parser.add_argument("-m", "--model_path", required=False,
                                    default='../model/trained_classifier_latest.pickle')
                parser.add_argument("-t", "--data_type", required=False, default=str(cam_parse),
                                    choices=["video", "folder", "webcam"])
                parser.add_argument("-p", "--data_path", required=False, default='data_openpose/' + str(file_path),
                                    help="path to a video file, or images folder, or webcam. \n"
                                         "For video and folder, the path should be "
                                         "absolute or relative to this project's root. "
                                         "For webcam, either input an index or device name. ")
                parser.add_argument("-o", "--output_folder", required=False, default='output/',
                                    help="Which folder to save result to.")

                args = parser.parse_args()
                return args

            args = parse_args()
            if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
                # If the path is not absolute, then its relative to the ROOT.
                args.data_path = ROOT + args.data_path
            return args

        def get_dst_folder_name(src_data_type, src_data_path):
            ''' 获取输出文件夹以及
            输出文件类型
            '''

            assert (src_data_type in ["video", "folder", "webcam"])

            if src_data_type == "video":  # /root/data/video.avi --> video
                folder_name = os.path.basename(src_data_path).split(".")[-2]

            elif src_data_type == "folder":  # /root/data/video/ --> video
                folder_name = src_data_path.rstrip("/").split("/")[-1]

            elif src_data_type == "webcam":
                # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
                folder_name = lib_commons.get_time_string()

            return folder_name

        # 将方法获取到的参数传输到args对象
        args = get_command_line_arguments()

        # 对每个对象值进行单独赋值
        SRC_DATA_TYPE = args.data_type
        SRC_DATA_PATH = args.data_path
        SRC_MODEL_PATH = args.model_path

        DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

        # -- configYAML文件中的参数

        cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
        cfg = cfg_all["s5_test.py"]

        CLASSES = np.array(cfg_all["classes"])
        SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

        # 压缩分辨率 以config里面的参数为准
        WINDOW_SIZE = int(cfg_all["features"]["window_size"])

        # 输出文件夹
        DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
        DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
        DST_VIDEO_NAME = cfg["output"]["video_name"]
        # 输出样例 video.avi
        DST_VIDEO_FPS = float(cfg["output"]["video_fps"])

        # 视频设置

        # 如果输入类型是Webcam的话需要在启动参数中指定maxframerate
        SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                                   ["webcam_max_framerate"])

        '''如果输入类型为视频，
        则需要设置每秒的ticks 
        例如3ticks / n 则每秒输出3帧
        '''

        SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                        ["video_sample_interval"])

        # Openpose 设置
        OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
        OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

        # Display 设置
        img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])

        # -- Function

        def select_images_loader(src_data_type, src_data_path):
            if src_data_type == "video":
                images_loader = lib_images_io.ReadFromVideo(
                    src_data_path,
                    sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

            elif src_data_type == "folder":
                images_loader = lib_images_io.ReadFromFolder(
                    folder_path=src_data_path)

            elif src_data_type == "webcam":
                if src_data_path == "":
                    webcam_idx = 0
                elif src_data_path.isdigit():
                    webcam_idx = int(src_data_path)
                else:
                    webcam_idx = src_data_path
                images_loader = lib_images_io.ReadFromWebcam(
                    SRC_WEBCAM_MAX_FPS, webcam_idx)
            return images_loader

        class MultiPersonClassifier(object):
            ''' 多人检测部分
            传统OPenpose只是识别姿态关键点
            如果是单人场景则没有问题
            但多人场景会出现关键点间连线问题
            这个时候如何对检测目标进行分类就很重要了
            具体实现方法：
            对检测到的每个人进行区域划分，命名为P(n)
            使用OpenPose只在其P0（示例）区域进行姿态点连线
            这样就不会出现姿态点乱连的问题了
            '''

            def __init__(self, model_path, classes):

                self.dict_id2clf = {}  # 检测目标ID -> 对每个人进行分类

                # 创建新的目标ID.
                self._create_classifier = lambda human_id: ClassifierOnlineTest(
                    model_path, classes, WINDOW_SIZE, human_id)

            def classify(self, dict_id2skeleton):
                ''' 对动作进行分类 '''

                # 清除不在检测范围内的目标
                old_ids = set(self.dict_id2clf)
                cur_ids = set(dict_id2skeleton)
                humans_not_in_view = list(old_ids - cur_ids)
                for human in humans_not_in_view:
                    del self.dict_id2clf[human]

                # 姿态估计
                id2label = {}
                for id, skeleton in dict_id2skeleton.items():

                    if id not in self.dict_id2clf:  # 添加新目标
                        self.dict_id2clf[id] = self._create_classifier(id)

                    classifier = self.dict_id2clf[id]
                    id2label[id] = classifier.predict(skeleton)  # 姿态label
                    # print("\n\nPredicting label for human{}".format(id))
                    # print("  skeleton: {}".format(skeleton))
                    # print("  label: {}".format(id2label[id]))

                return id2label

            def get_classifier(self, id):
                ''' 在划分的目标区域中进行姿态分类.
                Arguments:
                    id {int or "min"}
                '''
                if len(self.dict_id2clf) == 0:
                    return None
                if id == 'min':
                    id = min(self.dict_id2clf.keys())
                return self.dict_id2clf[id]

        def remove_skeletons_with_few_joints(skeletons):
            ''' 移除置信度较低（姿态关键点不完整）的骨架 '''
            good_skeletons = []
            for skeleton in skeletons:
                px = skeleton[2:2 + 13 * 2:2]
                py = skeleton[3:2 + 13 * 2:2]
                num_valid_joints = len([x for x in px if x != 0])
                num_leg_joints = len([x for x in px[-6:] if x != 0])
                total_size = max(py) - min(py)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
                    # add this skeleton only when all requirements are satisfied
                    good_skeletons.append(skeleton)
            return good_skeletons

        def draw_result_img(img_disp, ith_img, dict_id2skeleton, multiperson_classifier):
            ''' 绘制骨架，关键点和姿态lebel在检测区域内 '''

            # 图像归一处理
            r, c = img_disp.shape[0:2]
            desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
            img_disp = cv2.resize(img_disp,
                                  dsize=(desired_cols, img_disp_desired_rows))

            # 绘制所有人的骨架
            # skeleton_detector.draw(img_disp, humans)

            # 绘制目标检测区域内的骨架
            if len(dict_id2skeleton):
                for id, label in dict_id2label.items():
                    skeleton = dict_id2skeleton[id]
                    # scale the y data back to original
                    # 计算尺度系数
                    image_h = img_disp.shape[0]
                    image_w = img_disp.shape[1]
                    scale_h = 1.0 * image_h / image_w
                    skeleton[1::2] = skeleton[1::2] / scale_h
                    # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
                    lib_plot.draw_action_result(img_disp, id, skeleton, label)

            cv2.putText(img_disp, 'Frame:' + str(ith_img),
                        (20, 20), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(0, 0, 0), thickness=2)

            # 置信度（一个目标）
            if len(dict_id2skeleton):
                classifier_of_a_person = multiperson_classifier.get_classifier(id='min')
                classifier_of_a_person.draw_scores_onto_image(img_disp)
            return img_disp

        def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
            '''
            保存目标检测ID,姿态lebel，骨架信息
            In each image, for each skeleton, save the:
                human_id, label, and the skeleton positions of length 18*2.
            So the total length per row is 2+36=38
            '''
            skels_to_save = []
            for human_id in dict_id2skeleton.keys():
                label = dict_id2label[human_id]
                skeleton = dict_id2skeleton[human_id]
                skels_to_save.append([[human_id, label] + skeleton.tolist()])
            return skels_to_save

        # -- Main
        if __name__ == "__main__":

            # -- Detector, tracker, classifier
            # 1 注释掉
            # skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

            multiperson_tracker = Tracker()

            multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

            # -- Image reader and displayer
            images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)
            img_displayer = lib_images_io.ImageDisplayer()

            # -- 初始化输出

            # output folder
            os.makedirs(DST_FOLDER, exist_ok=True)
            os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

            # video writer
            video_writer = lib_images_io.VideoWriter(
                DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

            # -- Read images and process
            # 导入预训练模型
            estimator = load_pretrain_model('VGG_origin')
            try:
                ith_img = -1
                while images_loader.has_image():

                    # -- Read image
                    img = images_loader.read_image()
                    ith_img += 1
                    img_disp = img.copy()
                    print(f"\nProcessing {ith_img}th image ...")

                    # -- Detect skeletons
                    # humans = skeleton_detector.detect(img)
                    humans = estimator.inference(img)

                    image_height = img.shape[0]
                    image_width = img.shape[1]
                    scale_h = 1.0 * image_height / image_width

                    # skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
                    # skeletons = remove_skeletons_with_few_joints(skeletons)

                    # get pose info
                    # skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
                    skeletons = TfPoseVisualizer.draw_pose_rgb(img, humans)[1]  # return frame, joints, bboxes, xcenter

                    # 将骨架文件构造成对应的格式
                    new_format_skeletons = []
                    for person in skeletons:
                        this_person_skeleton = []
                        for i in range(18):
                            if i in person.keys():
                                this_person_skeleton.extend(
                                    [person[i][0] / image_width, scale_h * (person[i][1] / image_height)])
                            else:
                                this_person_skeleton.extend([0, 0])
                        new_format_skeletons.append(this_person_skeleton)
                    skeletons = new_format_skeletons

                    # 剔除部分节点
                    skeletons = remove_skeletons_with_few_joints(skeletons)

                    # 跟踪检测目标 每个人划分一个区域
                    dict_id2skeleton = multiperson_tracker.track(
                        skeletons)  # int id -> np.array() skeleton

                    # 识别每个人的动作
                    if len(dict_id2skeleton):
                        dict_id2label = multiperson_classifier.classify(
                            dict_id2skeleton)

                    # -- Draw
                    TfPoseVisualizer.draw_pose_rgb(img_disp, humans)

                    img_disp = draw_result_img(img_disp, ith_img, dict_id2skeleton, multiperson_classifier)

                    # Print label of a person
                    if len(dict_id2skeleton):
                        min_id = min(dict_id2skeleton.keys())
                        print("prediced label is :", dict_id2label[min_id])

                    # -- Display image, and write to video.avi
                    img_displayer.display(img_disp, wait_key_ms=1)
                    video_writer.write(img_disp)

                    # -- Get skeleton data and save to file
                    skels_to_save = get_the_skeleton_data_to_save_to_disk(
                        dict_id2skeleton)
                    lib_commons.save_listlist(
                        DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                        SKELETON_FILENAME_FORMAT.format(ith_img),
                        skels_to_save)
            finally:
                video_writer.stop()
                print("Openpose检测完成")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
