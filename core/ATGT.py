import sys        
sys.path.append('core/') 

import cv2  
import time 
import torch
import numpy as np
from sort.sort import Sort
from homography import homography as hg
from common import common
import threading
import datetime

from database.database import Database
from entity.vehicle import Criminal




class ATGT:

    def __init__(self, speed_estimate_area,tracking_area, deadline_main, estimateKM,max_speed_motocycle, max_speed_car, max_speed_truck,r):
        """
        Args:
            path_mode(string): path to model detect vehicle
            path_mode_license_plate(string): path to model detect license plate
            video_path(path): path to video demo
            speed_estimate_area(list): [[tl],[tr],[bl],[br]] ex:[[682,60],[912,76],[424,413],[1217,467]] this is distance in frame from camera,  get it to estimate speed
            tracking_area(list): [[tl],[tr],[bl],[br]] ex:[[682,60],[912,76],[424,413],[1217,467]] this is distance in frame from camera, get it to handle tracking
            deadline_main(list):  [[858,618],[2526,612]]. red line to detect blow the red light
            estimateKM(list or tuple ):(0.021,0.0156) (w,h)this is distance in real  world(km)
        """
        self.model =None


        self.CLS_MAPPING = {0:'motorcycle',1:'car',2:'truck'}
        self.COLOR = {'black': (0,0,0), 'white': (255,255,255),'green': (0, 199, 0), 'yellow':(253, 251, 37), 'red':(255,0,0)}
        self.fps = 0
        self.TrafficLight = common.TrafficLight( 'red', 15)
        self.stop = True
        self.criminals = []
        self.tracking_area = common.ChangePositionInList(np.multiply(tracking_area, r).astype(int), 2,3)
        self.speed_estimate_area =np.multiply(speed_estimate_area, r).astype(int)
        self.deadline_main = np.multiply(deadline_main,r).astype(int)
        self.estimateKM = estimateKM
        self.r = r
        self.inf = {}
        self.total = 0
        self.current_in_red_place = 0

        self.max_speed_motocycle  = max_speed_motocycle
        self.max_speed_car = max_speed_car
        self.max_speed_truck = max_speed_truck


        self.homography = hg.Homography(estimateKM, self.speed_estimate_area)
        self.mot_tracker = Sort(max_age=50, iou_threshold = 0.2, min_hits = 3, confidence_threshold = 0.5,homography = self.homography)
        self.db = Database()

        
        self.show_his = False
        self.show_bb = False
        self.show_handel_area = False
        self.show_deadline = False
        
    def init_model(self):
        print('================================START INITIALIZATION MODEL===================================')
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        # vehicle detection model 
        self.model = torch.hub.load('core/yolov5', 'custom', path='core/models/yolov5n_v2.pt', source='local').to(device)  # local repo
        self.model.eval()

        print('================================END INITIALIZATION MODEL===================================')




    def FrameByFrame(self,img, time_now):

        timestart_fps = time.time()

        source_img = img.copy()
        img = cv2.resize(img, (0,0), fx =self.r, fy = self.r)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        pre_det = self.model(img).pandas().xyxy[0].to_numpy()
        detections = np.array([box for box in pre_det if common.check_in_place(self.tracking_area,
                                (int(box[0]+(box[2]-box[0])/2),int(box[1]+(box[3]-box[1])/2)))])
        self.current_in_red_place = len(detections)
        trks = self.mot_tracker.update(common.convert_bbox_for_Sort(detections), time_now=time_now)

        for i in range(len(trks)):
            trk = trks[i]
            xmin, ymin, xmax, ymax = trk.get_state()[0]
            xmin, ymin, xmax, ymax =int(xmin), int(ymin), int(xmax), int(ymax)
            id_ = trk.id
            cls_ = trk.cls
            v = trk.estimate_speed

            if id_ not in self.inf: 
                self.inf[id_] = [cls_, common.random_color()]
            else:
                self.inf[id_][0] = cls_
            cx = int(xmin+(xmax-xmin)/2)
            cy = int(ymin+(ymax-ymin)/2)
                
            if len(trk.image) == 0:
                trk.image = source_img[int(ymin/self.r):int(ymax/self.r), int(xmin/self.r):int(xmax/self.r)]
                
            #handle when blow the red light
            where_now= common.CheckLine(self.deadline_main, [cx,cy])
            
            if (trk.where != 101) and (trk.where != where_now):
                if(self.TrafficLight.light=='red'):

                    criminal = Criminal(id_,'blow the red light', v, self.CLS_MAPPING[cls_], trk.image,trk.license_plate)
                    self.db.insert(criminal)

                if  (v>self.max_speed_car and cls_==1) or (v>self.max_speed_truck and cls_==2) or (v>self.max_speed_motocycle and cls_==0) :
                    criminal = Criminal(id_,'over speed', v, self.CLS_MAPPING[cls_], trk.image,trk.license_plate)
                    self.db.insert(criminal)

            trk.where = where_now
            ###################################SHOW INFORMATION##########################################
            if self.show_his:
                img = cv2.polylines(img, [np.array(trk.his_point.items, dtype=np.int32)], isClosed = False, color=self.inf[id_][1],thickness = 3)

            if self.show_bb:
                img = cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=self.inf[id_][1], thickness=1)
            
            img = cv2.putText(img, text=f'[ID: {id_} ][{self.CLS_MAPPING[cls_]}][{v:.2f}km/h]', org=(xmin, ymin-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.inf[id_][1], thickness=2)
            img = cv2.circle(img, (cx,cy), 3, self.inf[id_][1], thickness=-2)
            ############################################################################################

            
        ####################################SHOW GENERAL INFORMATION##########################################
        if self.show_handel_area:
            img = cv2.polylines(img, [self.tracking_area], True, self.COLOR['red'], thickness=1)
        if self.show_deadline:
            if self.TrafficLight.light == 'red':
                img = cv2.line(img, self.deadline_main[0],  self.deadline_main[1], color=self.COLOR['red'], thickness = 1)
            else:
                img = cv2.line(img, self.deadline_main[0],  self.deadline_main[1], color=self.COLOR['green'], thickness = 1)

        img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_shape = img.shape[:2]
        sub_img_shape = self.TrafficLight.img_light_traffic.shape[:2]
        img[:sub_img_shape[0], img_shape[1] - sub_img_shape[1]:,:] = self.TrafficLight.img_light_traffic
        #######################################################################################################

        self.fps = 1/(time.time()- timestart_fps)


        return img

    def thread_license_plate(self):
        index = 0
        while True:
            if index<len(self.criminals):
                img,plate = self.get_license_plate(self.criminals[index].image)
                print(plate)
                index+=1

    def process(self):
        cam = cv2.VideoCapture(self.video_path)
        if not cam.isOpened(): 
            self.TrafficLight.stop = True
            raise Exception('Could not open video')
        (major_ver, _, _) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
            print(f"Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {fps}")
        else :
            fps = cam.get(cv2.CAP_PROP_FPS)
            print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

        time_now = 0
        while True: 
            
            _, img = cam.read()
            #calc fps
            img = self.FrameByFrame(img, time_now)
            time_now+=1/fps
            cv2.imshow('source', img)
            if cv2.waitKey(1) == ord('q'):	
                self.TrafficLight.stop = True
                break
    def startThread(self):
        
        self.init_model()

        t1 = threading.Thread(target = self.TrafficLight.run, args = ())
        t2 = threading.Thread(target = self.process, args =())
        # t3 =  threading.Thread(target = self.thread_license_plate, args =())
        t1.start()
        t2.start()
        # t3.start()
        t1.join()
        t2.join()
        # t3.join()




# speed_estimate_area = [[1068,204],[2325,219],[171,1741],[3206,1759]]#[[213,124],[364,125],[7,399],[361,404]]#[[350,49],[577,59],[317,454],[807,456]]#
# tracking_area = [[888,424],[1265,422],[820,1011],[1567,1005]]
# deadline = [[858,618],[2526,612]]
# estimateKM = (0.0156,0.021)#(0.021,0.0156)#(w,h)(0.0215 ,0.00989)
# video_path = r'E:\IAMDAT\Data\DOANCUOIKY\Video\91.mp4'
# path_model = 'core/models/yolov5n_v2.pt'
# path_detect_number = 'core/models/detect_number_2.pt'
# path_mode_license_plate = 'core/models/best_lp_4.pt'
# atgt = ATGT(path_model, path_mode_license_plate,path_detect_number ,video_path, speed_estimate_area,speed_estimate_area, deadline, estimateKM,0.3)
# # atgt.run(video_path, speed_estimate_area,speed_estimate_area, estimateKM,0.4)
# atgt.startThread()

