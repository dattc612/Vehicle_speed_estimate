import math
import time
from threading import Thread
import cv2
import numpy as np

####################################################################################
COLOR = {'black': (0,0,0), 'white': (255,255,255),'green': (0, 199, 0), 'yellow':(253, 251, 37), 'red':(255,0,0)}
###################################################################################
def EstimationSpeed(previous_point, current_point, hidden_time):
		"""
		kc: float(m)

		"""
		px,py = previous_point
		qx,qy = current_point
		#kc = 2*math.asin(math.sqrt(math.sin((px-qx)/2)**2 +math.cos(px)* math.cos(qx)*(math.sin(py-qy)/2)**2))
		
		kc = math.sqrt((qx-px)**2+(qy-py)**2)
		v = kc/(hidden_time/30/3600)
		return v

def CheckLine(deadline, w):
	p1 = deadline[0]
	p2 = deadline[1]
	x = (p2[0]-p1[0])*(w[1]-p1[1])-(w[0]-p1[0])*(p2[1]-p1[1])
	if x<0:
		return -1
	elif x>=0:
		return 1
	else:
		return 0  	

def check_in_place(area, pt):
	return cv2.pointPolygonTest(np.array(area, dtype=np.int64).reshape(1,-1,2), pt, False)>=0

def random_color():
	return list(np.random.random(size=3) * 256)

def convert_bbox_for_Sort(boxes):
    		# raw box (dataframe) 
		if len(boxes) ==0:
			return np.empty((0,6))
		bboxes = np.empty((len(boxes), 6))
		bboxes[...,0] = boxes[..., 0]
		bboxes[...,1] = boxes[..., 1] 
		bboxes[...,2] = boxes[..., 2] 	
		bboxes[...,3] = boxes[..., 3] 
		bboxes[...,4] = boxes[...,5]
		bboxes[...,5] = boxes[..., 4]
		return bboxes

#################################################################################

class Queue:
	def __init__(self, length = 20):
		self.length = length
		self.items = []
	def update(self,item):
		if len(self.items) <= self.length:
			self.items.append(item)
		else:
			self.items.pop(0)
			self.items.append(item)
	def delete(self):
		self.items.pop(0)


class TrafficLight(Thread):
	def __init__(self,light = 'green', time_each_loop = 30):
		"""
		start_light: string(red or green)
		time_each_loop: time to change light
		"""
		self.light = light
		self.time_each_loop = time_each_loop
		self.stop = False
	def run(self):
		while(not self.stop):
			time.sleep(self.time_each_loop)
			if self.light == 'red':
				self.light = 'green'
			else:
				self.light = 'red'

class Criminal:
	def __init__(self, id, type_criminal, images):
		self.id = id
		self.type_criminal = type_criminal
		self.images = images
