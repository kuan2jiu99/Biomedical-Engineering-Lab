import sys
import cv2
import csv
import time
import threading
import serial  # 引用pySerial模組
import datetime
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from scipy import signal
import mediapipe as mp
from PIL import Image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

RIGHT_MAX = 800
RIGHT_MIN = 0
LEFT_MAX = 800
LEFT_MIN = 0
# RIGHT_MAX = 0
# RIGHT_MIN = 1000
# LEFT_MAX = 0
# LEFT_MIN = 1000

MAX = 500
window_size = 10
window_size_stft = 100

# for fatigue
fs = 480.0  # Sample frequency (Hz)
f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
showlist = [
    "raw", 
    "balance", 
    "fatigue", 
    # "balance_graph",
]

# show = "硬舉正面"
# show = "硬舉側面"
show = "深蹲正面"
# show = "深蹲側面"

x = np.array(range(MAX))
y1 = np.array([0 for i in range(MAX)])
y2 = np.array([0 for i in range(MAX)])
y1_amp = np.array([0 for i in range(MAX // window_size)])
y2_amp = np.array([0 for i in range(MAX // window_size)])
y1_fat = np.array([0 for i in range(MAX // window_size_stft * 3)])
y2_fat = np.array([0 for i in range(MAX // window_size_stft * 3)])
fatigue_max1 = 0
fatigue_min1 = 0
fatigue_max2 = 0
fatigue_min2 = 0
ratio = 0

def emg_utils(right_raw_array, left_raw_array, threshold=0.45):
    global RIGHT_MAX, RIGHT_MIN, LEFT_MAX, LEFT_MIN
    # Input: 
    # (1) right raw emg data. 1
    # (2) left raw emg data. 2

    # Output:
    # (1) processed right emg data.
    # (2) processed left emg data.
    # (3) whether it is balance, true for balance, false for unbalanced.

    # get window size.
    window_size = right_raw_array.shape[0]

    # if np.max(right_raw_array) > RIGHT_MAX:
    #     RIGHT_MAX = np.max(right_raw_array)
    # if np.max(left_raw_array) > LEFT_MAX:
    #     LEFT_MAX = np.max(left_raw_array)

    # if np.min(right_raw_array) < RIGHT_MIN:
    #     RIGHT_MIN = np.min(right_raw_array)
    # if np.min(left_raw_array) < LEFT_MIN:
    #     LEFT_MIN = np.min(left_raw_array)

    # normalization.
    right_norm_array = (right_raw_array - RIGHT_MAX) / (RIGHT_MAX - RIGHT_MIN)
    left_norm_array = (left_raw_array - LEFT_MAX) / (1.4*(LEFT_MAX - LEFT_MIN))

    # p2p value of right.
    peak_max = max(right_norm_array)
    peak_min = min(right_norm_array)
    peak2peak_right = peak_max - peak_min

    # p2p value of left.
    peak_max = max(left_norm_array)
    peak_min = min(left_norm_array)
    peak2peak_left = peak_max - peak_min

    # print(peak2peak_right, peak2peak_left)

    if abs(peak2peak_right - peak2peak_left) < threshold:
        is_balanced = True
    else:
        is_balanced = False


    return peak2peak_right, peak2peak_left, is_balanced

weighted_prev = 0
Zxx_prev = -1
def fatigue(y):
    global weighted_prev, Zxx_prev
    y = np.array(y) - np.mean(y)
    f, t, Zxx = signal.stft(y, fs, nperseg=window_size_stft)
    Zxx_max = f[np.argmax(np.abs(Zxx)[1:], axis=0)]
    Zxx_max_edited = []
    for i in range(len(Zxx_max)):
        if Zxx_max[i] > 50:
            Zxx_max_edited.append(Zxx_prev)
        else:
            Zxx_max_edited.append(Zxx_max[i])
            Zxx_prev = Zxx_max[i]

    Zxx_max_ave = []
    for i in range(len(Zxx_max_edited)):
        weighted_prev = weighted_prev * 0.97 + Zxx_max_edited[i] * 0.03
        Zxx_max_ave.append(weighted_prev)
    return f, Zxx_max_ave

class RectItem(pg.GraphicsObject):
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._generate_picture(color="g")

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self, color):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen("w"))
        painter.setBrush(pg.mkBrush(color))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

def main1():
    global y1, y2, y1_amp, y2_amp, y1_fat, y2_fat
    global window_size, window_size_stft
    global fatigue_max1, fatigue_max2
    global fatigue_min1, fatigue_min2
    global ratio
    global showlist
    class App(QtGui.QMainWindow):
        def __init__(self, parent=None):
            super(App, self).__init__(parent)

            #### Create Gui Elements ###########
            self.mainbox = QtGui.QWidget()
            self.setCentralWidget(self.mainbox)
            self.mainbox.setLayout(QtGui.QVBoxLayout())

            self.canvas = pg.GraphicsLayoutWidget()
            self.mainbox.layout().addWidget(self.canvas)

            self.label = QtGui.QLabel()
            self.mainbox.layout().addWidget(self.label)

            # self.view = self.canvas.addViewBox()
            # self.view.setAspectLocked(True)
            # self.view.setRange(QtCore.QRectF(0,0, 100, 100))

            #  image plot
            # self.img = pg.ImageItem(border='w')
            # self.view.addItem(self.img)

            #  line plot
            if "raw" in showlist:
                self.otherplot = self.canvas.addPlot()
                self.otherplot.vb.setYRange(-100, 1124)
                self.h1 = self.otherplot.plot(pen='y')
                self.canvas.nextRow()
                self.otherplot = self.canvas.addPlot()
                self.otherplot.vb.setYRange(-100, 1124)
                self.h2 = self.otherplot.plot(pen='y')
            if "balance" in showlist:
                self.canvas.nextRow()
                self.otherplot = self.canvas.addPlot()
                self.otherplot.vb.setYRange(0, 1.5)
                self.h3 = self.otherplot.plot(pen='g')
                self.h4 = self.otherplot.plot(pen='r')
            if "fatigue" in showlist:
                self.canvas.nextRow()
                self.otherplot = self.canvas.addPlot()
                self.otherplot.vb.setYRange(0, 50)
                self.h5 = self.otherplot.plot(pen='b')
                # self.h5_max = self.otherplot.plot(pen='b')
                # self.h5_min = self.otherplot.plot(pen='b')
                self.canvas.nextRow()
                self.otherplot = self.canvas.addPlot()
                self.otherplot.vb.setYRange(0, 50)
                self.h6 = self.otherplot.plot(pen='b')
                # self.h6_max = self.otherplot.plot(pen='b')
                # self.h6_min = self.otherplot.plot(pen='b')
            if "balance_graph" in showlist:
                self.canvas.nextRow()
                self.view = self.canvas.addViewBox()
                rect_item = RectItem(QtCore.QRectF(0.5, 0, 0.1, 0.5))
                self.view.addItem(rect_item)

            #### Set Data  #####################

            # self.x = np.linspace(0,50., num=100)
            # self.X,self.Y = np.meshgrid(self.x,self.x)

            self.counter = 0
            self.fps = 0.
            self.lastupdate = time.time()

            #### Start  #####################
            self._update()

        def _update(self):

            # self.data = np.sin(self.X/3.+self.counter/9.)*np.cos(self.Y/3.+self.counter/9.)
            # self.ydata = y2#np.sin(self.x/3.+ self.counter/9.)

            # self.img.setImage(self.data)
            if "raw" in showlist:
                self.h1.setData(y1)
                self.h2.setData(y2)
            if "balance" in showlist:
                self.h3.setData(y1_amp)
                self.h4.setData(y2_amp)
            if "fatigue" in showlist:
                self.h5.setData(y1_fat)
                # self.h5_max.setData(np.repeat(fatigue_max1, y1_fat.shape[0]))
                # self.h5_min.setData(np.repeat(fatigue_min1, y1_fat.shape[0]))
                self.h6.setData(y2_fat)
                # self.h6_max.setData(np.repeat(fatigue_max2, y2_fat.shape[0]))
                # self.h6_min.setData(np.repeat(fatigue_min2, y2_fat.shape[0]))
            # Update the size of the rectangle
            if "balance_graph" in showlist:
                if ratio > 0:
                    new_rect = QtCore.QRectF(0.5, 0, 1 * ratio, 0.5)
                else:
                    new_rect = QtCore.QRectF(0.5 + 1 * ratio, 0, -1 * ratio, 0.5)
                # print(ratio)
                self.view.addedItems[0]._rect = new_rect
                if abs(ratio) > 0.2:
                    self.view.addedItems[0]._generate_picture(color="r")
                else:
                    self.view.addedItems[0]._generate_picture(color="g")
                # # Redraw the rectangle in the graphics scene
                self.view.addedItems[0].update()
                # self.view.addedItems[0]._rect.setRect(0, 0, 1, 0.1)
                # self.view.addedItems[0].update()

            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
            self.label.setText(tx)
            QtCore.QTimer.singleShot(1, self._update)
            self.counter += 1

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())

def main2_csv():
    global y1, y2, y1_amp, y2_amp, y1_fat, y2_fat
    global window_size, window_size_stft
    global fatigue_max1, fatigue_max2
    global fatigue_min1, fatigue_min2
    global ratio
    delay_time = (1 / 480) * window_size
    f = open('12-05_3重心不穩.csv', 'r', newline='')
    # f = open('12-05_6fatigue.csv', 'r', newline='')
    rows = csv.reader(f)
    window_y1 = []
    window_y2 = []
    window_y1_fatigue = []
    window_y2_fatigue = []
    i = 0
    for row in rows:
        # t.append(int(row[0]))
        window_y1.append(int(row[2]))
        window_y2.append(int(row[3]))
        window_y1_fatigue.append(int(row[2]))
        window_y2_fatigue.append(int(row[3]))
        
        if i % window_size == window_size-1:
            y1 = np.concatenate((y1, window_y1))[window_size:]
            y2 = np.concatenate((y2, window_y2))[window_size:]
            window_y1_amp, window_y2_amp, _ = emg_utils(np.array(window_y1), np.array(window_y2))
            ratio = 0.9 * ratio + 0.1 * -1 * (window_y1_amp - window_y2_amp)
            y1_amp = np.concatenate((y1_amp, [window_y1_amp]))[1:]
            y2_amp = np.concatenate((y2_amp, [window_y2_amp]))[1:]
            window_y1 = []
            window_y2 = []
            time.sleep(delay_time)

        if i % window_size_stft == window_size_stft-1:
            f1, z1 = fatigue(window_y1_fatigue)
            f2, z2 = fatigue(window_y2_fatigue)
            y1_fat = np.concatenate((y1_fat, z1))[3:]
            y2_fat = np.concatenate((y2_fat, z2))[3:]
            window_y1_fatigue = []
            window_y2_fatigue = []
            fatigue_max1 = np.max(y1_fat)
            fatigue_min1 = np.min(y1_fat)
            fatigue_max2 = np.max(y2_fat)
            fatigue_min2 = np.min(y2_fat)
        i += 1

def main2_real():
    write = False ##################################################################################
    global y1, y2, y1_amp, y2_amp, y1_fat, y2_fat
    global window_size, window_size_stft
    global fatigue_max1, fatigue_max2
    global fatigue_min1, fatigue_min2
    global ratio
    COM_PORT = 'COM3'    # 指定通訊埠名稱
    BAUD_RATES = 38400    # 設定傳輸速率
    ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠
    window_y1 = []
    window_y2 = []
    window_y1_fatigue = []
    window_y2_fatigue = []
    i = 0

    # if write:
    #     f = open('12-07_6深蹲.csv', 'w', newline='')
    #     writer = csv.writer(f)
    while True:
        while ser.in_waiting:          # 若收到序列資料…
            data_raw = ser.readline()  # 讀取一行
            # data = data_raw.decode()   # 用預設的UTF-8解碼
            # print(data_raw)
            try: 
                data = int(data_raw)
            except ValueError:
                continue
            data1 = data // 1024
            data2 = data % 1024
            # if write:
            #     writer.writerow([i, datetime.datetime.now(), data1[0], data2[0]])
            # print('i: {:6d}, data1: {:4d}, data2: {:4d}, time: {:s}'.format(i, data1[0], data2[0], str(datetime.datetime.now().time()).replace(':', '-')))
            # y1 = np.concatenate((y1, data1))[1:]
            # y2 = np.concatenate((y2, data2))[1:]
            window_y1.append(data1)
            window_y2.append(data2)
            window_y1_fatigue.append(data1)
            window_y2_fatigue.append(data2)
            
            if i % window_size == window_size-1:
                y1 = np.concatenate((y1, window_y1))[window_size:]
                y2 = np.concatenate((y2, window_y2))[window_size:]
                window_y1_amp, window_y2_amp, _ = emg_utils(np.array(window_y1), np.array(window_y2))
                ratio = 0.9 * ratio + 0.1 * -1 * (window_y1_amp - window_y2_amp)
                y1_amp = np.concatenate((y1_amp, [window_y1_amp]))[1:]
                y2_amp = np.concatenate((y2_amp, [window_y2_amp]))[1:]
                window_y1 = []
                window_y2 = []
                # time.sleep(delay_time)

            if i % window_size_stft == window_size_stft-1:
                f1, z1 = fatigue(window_y1_fatigue)
                f2, z2 = fatigue(window_y2_fatigue)
                y1_fat = np.concatenate((y1_fat, z1))[3:]
                y2_fat = np.concatenate((y2_fat, z2))[3:]
                window_y1_fatigue = []
                window_y2_fatigue = []
                fatigue_max1 = np.max(y1_fat)
                fatigue_min1 = np.min(y1_fat)
                fatigue_max2 = np.max(y2_fat)
                fatigue_min2 = np.min(y2_fat)
            i += 1 
            # print(i)

def main3():
    fps = 20
    frameWidth  = 640
    frameHeight = 360
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_FPS, fps)

    cameraFPS = cap.get(cv2.CAP_PROP_FPS)

    print("FPS:", cameraFPS)
    print("Frame size:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap = cv2.VideoCapture("屁股眨眼+下背壓力校正.mov")
    with mp_pose.Pose(
        enable_segmentation = True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            ret1, origin = success, image
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.flip(image, -1)
            results = pose.process(image)
            # print(type(results.segmentation_mask))
            # origin = cv2.flip(origin, -1)
            # image = cv2.flip(image, -1)
            if type(results.segmentation_mask) == type(np.array([1])):
                # SEGMENTATION_MASK
                BG_COLOR = (255,255,255) 
                # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                # bg_image = np.zeros(image.shape, dtype=np.uint8)
                # bg_image[:] = BG_COLOR
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = np.where(condition, image, bg_image)


            

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Flip the image horizontally for a selfie-view display.
            # origin = cv2.flip(origin, -1)
            image = cv2.flip(image, -1)
            image = np.concatenate([image, np.full_like(image, 255)], axis=1)
            # image = np.concatenate([origin, image], axis=1)
            if type(results.segmentation_mask) == type(np.array([1])):
                # 骨盆眨眼
                if show == "深蹲側面":
                    if get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")[1] < get_landmark(results.pose_landmarks.landmark, "RIGHT_KNEE")[1]:
                        cv2.putText(image, f'Hip : Please go up', (360, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        # print('Hip : Please go up')
                    else:
                        cv2.putText(image, f'Hip : Good', (360, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
                        # print('Hip : Good')
                    # cv2.putText(image, f'KNEE : {round(get_landmark(results.pose_landmarks.landmark, "RIGHT_KNEE")[1],3)}', (360, 150), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0.8, 0, 0), 1, cv2.LINE_AA)
                
                # 重心歪向一邊
                if show == "深蹲正面":
                    middle_p(results) >=0.042
                    q = get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")[0]
                    w = get_landmark(results.pose_landmarks.landmark, "LEFT_HIP")[0]
                    e = get_landmark(results.pose_landmarks.landmark, "RIGHT_ANKLE")[0]
                    r = get_landmark(results.pose_landmarks.landmark, "LEFT_ANKLE")[0]
                    t = round(abs(q-e),3)
                    y = round(abs(w-r),3)
                    if  ((y-t>0.08)):
                        cv2.putText(image, f'Balance : more left foot', (360, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    elif ((t-y>0.08)):
                        cv2.putText(image, f'Balance : more right foot', (360, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Balance : Good', (360, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)    

                # 膝蓋內夾
                if show == "深蹲正面":
                    knee_angles = get_knee_angle(results.pose_landmarks.landmark)
                    if (knee_angles[0] < 178) & (get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")[1] < 0.7) & (get_landmark(results.pose_landmarks.landmark, "RIGHT_KNEE")[0] < get_landmark(results.pose_landmarks.landmark, "RIGHT_ANKLE")[0]) & (get_landmark(results.pose_landmarks.landmark, "LEFT_KNEE")[0] > get_landmark(results.pose_landmarks.landmark, "LEFT_ANKLE")[0]):
                        cv2.putText(image, f'Knee : Knee valgus', (360, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Knee : Good', (360, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

                # 下背壓力   
                if show == "深蹲側面":
                    if if_eq_slope(results):
                        cv2.putText(image, f'Back : Good', (360, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Back : Under pressure', (360, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    

                    

                # 硬舉部分 -----------------------------------------------------------
                # 左右腳出力不平均
                if show == "硬舉正面":
                    if (get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")[1] < get_landmark(results.pose_landmarks.landmark, "LEFT_SHOULDER")[1]) & (get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")[1] < get_landmark(results.pose_landmarks.landmark, "LEFT_HIP")[1]) | (middle_p(results) >=0.1):
                        cv2.putText(image, f'Balance : more left foot', (360, 350), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    elif (get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")[1] > get_landmark(results.pose_landmarks.landmark, "LEFT_SHOULDER")[1]) & (get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")[1] > get_landmark(results.pose_landmarks.landmark, "LEFT_HIP")[1]) | (middle_p(results) <=-0.1):
                        cv2.putText(image, f'Balance : more right foot', (360, 350), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Balance : Good', (360, 350), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)   
                
                # 站立時過度挺腰
                if show == "硬舉側面":
                    if (get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")[0] > get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")[0]+0.05) | (get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")[0] > get_landmark(results.pose_landmarks.landmark, "RIGHT_ANKLE")[0]+0.05):
                        cv2.putText(image, f'Standing : Bend too far', (360, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Standing : Good', (360, 400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
                
                # 駝背
                if show == "硬舉側面":
                    a ,b = mark_middle(results, image)
                    if b >= 846:
                        cv2.putText(image, f'Back : Under pressure',(360, 450) ,cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Back : Good', (360, 450), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
                
                # 抬頭
                if show == "硬舉側面":
                    if (get_landmark(results.pose_landmarks.landmark, "NOSE")[1] > get_landmark(results.pose_landmarks.landmark, "RIGHT_EAR")[1]):
                        cv2.putText(image, f'Neck : Bend too far', (360, 500), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(image, f'Neck : Good', (360, 500), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, f'People not detected', (360, 350), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)   
            
            image = cv2.resize(image, (1080, 960), interpolation=cv2.INTER_AREA)
            cv2.imshow('MediaPipe Pose', image)
            
            # videofile.write(image)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()

# Function to get landmarks
def get_landmark(landmarks, part_name):
    return [
        landmarks[mp_pose.PoseLandmark[part_name].value].x,
        landmarks[mp_pose.PoseLandmark[part_name].value].y,
        landmarks[mp_pose.PoseLandmark[part_name].value].z,
    ]
    
def if_eq_slope(results):
    temp1 = get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")
    temp2 = get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")
    temp3 = get_landmark(results.pose_landmarks.landmark, "RIGHT_KNEE")
    temp4 = get_landmark(results.pose_landmarks.landmark, "RIGHT_ANKLE")
    slope1 = (temp1[0] - temp2[0]) / (temp1[1] - temp2[1])
    slope2 = (temp3[0] - temp4[0]) / (temp3[1] - temp4[1])
    return abs(slope1 - slope2) < 0.4

def middle_p(results):
    temp1 = get_landmark(results.pose_landmarks.landmark, "LEFT_HIP")
    temp2 = get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")
    temp3 = get_landmark(results.pose_landmarks.landmark, "LEFT_KNEE")
    temp4 = get_landmark(results.pose_landmarks.landmark, "RIGHT_ANKLE")
    
    mid1 = (temp1[0] +temp2[0]) / 2
    mid2 = (temp3[0] +temp4[0]) / 2
    
    return (round(mid1 - mid2,4))
    
def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def mark_middle(results,image):
    
    temp1 = get_landmark(results.pose_landmarks.landmark, "RIGHT_HIP")
    temp0 = get_landmark(results.pose_landmarks.landmark, "RIGHT_SHOULDER")
    
    slope = (temp0[1] - temp1[1]) / (temp0[0] - temp1[0])
    slope_v = -1 / slope
    avg_x = ( temp0[0] + temp1[0] ) / 2
    avg_y = ( temp0[1] + temp1[1] ) / 2
    

    
    ind = 0
    step = 0.01
    p_x = int(round(avg_x,2)*1125) - 50
    p_y = 1200-int(round(avg_y,2)* 900)
    
    return [p_x,p_y]
    
def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    m_hip = (r_hip + l_hip)
    m_hip = [x / 2 for x in m_hip]
    m_knee = (r_knee + l_knee)
    m_knee = [x / 2 for x in m_knee]
    m_ankle = (r_ankle + l_ankle)
    m_ankle = [x / 2 for x in m_ankle]

    mid_angle = calc_angles(m_hip, m_knee, m_ankle)

    return [r_angle, l_angle, mid_angle]

# 定義線程
t_list = []

t1 = threading.Thread(target=main1, args=())
t_list.append(t1)
t2 = threading.Thread(target=main2_real, args=())
t_list.append(t2)
t3 = threading.Thread(target=main3, args=())
t_list.append(t3)

# 開始工作
for t in t_list:
    t.start()
 
# 調整多程順序
for t in t_list:
    t.join()