from threading import Thread
import cv2
class WebcamVideoStream:
    def __init__(self, src=0):
        #初始化攝像機並從中讀取第一幀
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        #初始化用於指示線程是否應該停止的變量
        self.stopped = False
    def start(self):
        #啟動線程從影像中讀取幀
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        #繼續無限循環, 直到線程停止
        while True:
            #如果設置了線程指示器變量, 則停止線程
            if self.stopped:
                return
            #否則, 從中讀取下一幀
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        #返回最近讀取的幀
        return self.frame
    def stop(self):
        #表示應該停止線程
        self.stopped = True
    def isOpened(self):
        return self.isOpened()