import cv2  
import time 
import torch
import sys
import numpy as np
from sort.sort import Sort
from homography import homography as hg
from PIL import Image
from common import common
import threading
import logging



class ATGT:
	def __init__(self,path_model,path_mode_license_plate):
		self.path_model = path_model
		self.path_mode_license_plate = path_mode_license_plate
		self.model =None
		self.licensePlateModel =None
		self.homography = None
		self.homography_mapping_mini_image = None
		self.handle_size = (500,1000)
		self.mot_tracker = None
		self.CLS_MAPPING = {0:'motorcycle',1:'car',2:'truck'}
		self.COLOR = {'black': (0,0,0), 'white': (255,255,255),'green': (0, 199, 0), 'yellow':(253, 251, 37), 'red':(255,0,0)}
		self.fps = 0
		self.TrafficLight = common.TrafficLight( 'red', 15)
		self.stop = True
		self.criminals = []
		
	def init_model(self, path_model_1, path_model_2):
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

		# vehicle detection model 
		logging.info('START INITIALIZE MODEL DETECT VEHICLE')
		self.model = torch.hub.load('yolov5', 'custom', path=path_model_1, source='local').to(device)  # local repo
		self.model.eval()
		logging.info('END INITIALIZE MODEL DETECT VEHICLE')

		# init license plate model
		logging.info('START INITIALIZE MODEL DETECT LICENSE PLATE')
		self.licensePlateModel = torch.hub.load('yolov5', 'custom', path=path_model_2, source='local').to(device)  # local repo
		self.licensePlateModel.eval()
		logging.info('END INITIALIZE MODEL DETECT LICENSE PLATE')

		
	
	

	def getLicensePlate(self,sub_img):
		return sub_img
	def process(self, video_path,speed_estimate_area,tracking_area, deadline,estimateKM,r = 1):
		"""
		haddle_area: list  [[tl],[tr],[bl],[br]] ex:[[682,60],[912,76],[424,413],[1217,467]]
		estimateKM: tuple (w,h)
		video_path: string
		"""
		speed_estimate_area =np.multiply(speed_estimate_area, r).astype(int)
		deadline = np.multiply(deadline,r).astype(int)
		# width,height of mini map
		(h,w) = estimateKM
		self.homography = hg.Homography((w,h), speed_estimate_area)
		self.mot_tracker = Sort(max_age=50, iou_threshold = 0.2, min_hits = 3, confidence_threshold = 0.5,homography = self.homography)

		cam = cv2.VideoCapture(video_path)
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
		
		inf = {}

		while True: 
			timestart_fps = time.time()
			
			_, img = cam.read()
			source_img = img.copy()

			img = cv2.resize(img, (0,0), fx =r, fy = r)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


			#draw area detection
			pts = np.multiply(tracking_area, r).astype(int)
			temp = pts[3].copy()
			pts[3] = pts[2]
			pts[2] = temp
			

			#predict bbox format (xmin ymin xmax ymax)
			pre_det = self.model(img).pandas().xyxy[0].to_numpy()

			detections = np.array([box for box in pre_det if common.check_in_place(pts,
								(int(box[0]+(box[2]-box[0])/2),int(box[1]+(box[3]-box[1])/2)))])

			trks = self.mot_tracker.update(common.convert_bbox_for_Sort(detections), fps = self.fps)
			for i in range(len(trks)):
				trk = trks[i]
				xmin, ymin, xmax, ymax = trk.get_state()[0]
				xmin, ymin, xmax, ymax =int(xmin), int(ymin), int(xmax), int(ymax)
				id_ = trk.id
				cls_ = trk.cls
				v = trk.estimate_speed

				if id_ not in inf: 
					inf[id_] = [cls_, common.random_color()]
				else:
					inf[id_][0] = cls_
				cx = int(xmin+(xmax-xmin)/2)
				cy = int(ymin+(ymax-ymin)/2)
				
				
				if len(trk.image) == 0:
					trk.image = source_img[int(ymin/r):int(ymax/r), int(xmin/r):int(xmax/r)]

				#handle when pass deadline
				where_now= common.CheckLine(deadline, [cx,cy])
				if (trk.where != 101) and (trk.where != where_now) and (self.TrafficLight.light=='red'):
					self.criminals.append(common.Criminal(trk.id, 'Blow the red light', trk.image)) 
					# cv2.imwrite('images/sub_images/'+str(trk.id)+'.jpg', trk.image)
					cv2.imshow('hi', trk.image)
				trk.where = where_now
				
				###################################show infomation##########################################
				img = cv2.polylines(img, [np.array(trk.his_point.items, dtype=np.int32)], isClosed = False, color=inf[id_][1],thickness = 3)
				img = cv2.putText(img, text=f'[ID: {id_} ][{self.CLS_MAPPING[cls_]}][{v:.2f}km/h]', org=(xmin, ymin-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=inf[id_][1], thickness=2)
				# img = cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=inf[id_][1], thickness=1)
				img = cv2.circle(img, (cx,cy), 3, inf[id_][1], thickness=-2)
				############################################################################################

			
			####################################show general infomation##########################################
			# img = cv2.polylines(img, [pts], True, self.COLOR['red'], thickness=1)
			if self.TrafficLight.light == 'red':
				img = cv2.line(img, deadline[0],  deadline[1], color=self.COLOR['red'], thickness = 1)
			else:
				img = cv2.line(img, deadline[0],  deadline[1], color=self.COLOR['green'], thickness = 1)
			img = cv2.putText(img, text = f'fps: {self.fps}',org = (0,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5,color=self.COLOR['red'], thickness=2)
			img = cv2.putText(img, text = 'in red place: ' + str(len(detections)),org = (0,60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5,color=self.COLOR['red'], thickness=2)
			img = cv2.putText(img, text = 'total: ' + str(len(inf)),org = (0,90), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5,color=self.COLOR['red'], thickness=2)
			img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			#######################################################################################################
			#calc fps
			self.fps = 1/(time.time()- timestart_fps)

			cv2.imshow('source', img)
			if cv2.waitKey(1) == ord('q'):	
				self.TrafficLight.stop = True
				break


	def startThread(self, video_path,speed_estimate_area,tracking_area, deadline,estimateKM,r = 1):
		
		self.init_model(self.path_model,self.path_mode_license_plate)

		t1 = threading.Thread(target = self.TrafficLight.run, args = ())
		t2 = threading.Thread(target = self.process, args =(video_path,speed_estimate_area,tracking_area, deadline,estimateKM, r))
		t1.start()
		t2.start()
		t1.join()
		t2.join()




speed_estimate_area = [[1068,204],[2325,219],[171,1741],[3206,1759]]#[[213,124],[364,125],[7,399],[361,404]]#[[350,49],[577,59],[317,454],[807,456]]#
tracking_area = [[888,424],[1265,422],[820,1011],[1567,1005]]
deadline = [[858,618],[2526,612]]
estimateKM = (0.021,0.0156)#(0.0215 ,0.00989)
video_path = r'E:\IAMDAT\Data\DOANCUOIKY\Video\91.mp4'
path_model = 'models/best2.pt'
path_mode_license_plate = 'models/best_lp.pt'
atgt = ATGT(path_model, path_mode_license_plate)
# atgt.run(video_path, speed_estimate_area,speed_estimate_area, estimateKM,0.4)
atgt.startThread(video_path, speed_estimate_area,speed_estimate_area, deadline, estimateKM,0.3)





# import json 

# with open('config/cfg.json', 'r') as f:
#   data = json.load(f)

# print(type(data['speed']['estimateKM']))

