"""
  @Author: hawcat
  @Date  : 2023/3/16 20:43
  @version V1.1
"""
#导入开源Tfpose
from pose_visualizer import TfPoseVisualizer
import numpy as np
import cv2
from pathlib import Path
import argparse

if True:  # Include project path
    import sys
    import os
    import PIL.Image
    import PIL.ImageDraw
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
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


# -- Command-line input
def get_command_line_arguments():
#参数
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='../model/trained_classifier_latest.pickle')
        parser.add_argument("-t",    "--data_type", required=False, default='video',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="data_openpose/test.mp4",
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

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()

    return folder_name

#将方法获取到的参数传输到args对象
args = get_command_line_arguments()

#对每个对象值进行单独赋值
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
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
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



def draw_result_img(img_disp, ith_img, dict_id2skeleton,multiperson_classifier):
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
            image_h=img_disp.shape[0]
            image_w=img_disp.shape[1]
            scale_h=1.0 * image_h / image_w
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            lib_plot.draw_action_result(img_disp, id, skeleton, label)


    cv2.putText(img_disp, 'Frame:'+ str(ith_img) ,
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
            new_format_skeletons=[]
            for person in skeletons:
                this_person_skeleton=[]
                for i in range(18):
                    if i in person.keys():
                        this_person_skeleton.extend([person[i][0]/image_width,scale_h*(person[i][1]/image_height)])
                    else:
                        this_person_skeleton.extend([0,0])
                new_format_skeletons.append(this_person_skeleton)
            skeletons=new_format_skeletons

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

            img_disp = draw_result_img(img_disp, ith_img, dict_id2skeleton,multiperson_classifier)

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
        print("Program ends")
