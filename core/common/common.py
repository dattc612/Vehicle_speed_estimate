import math
import time
from threading import Thread
import cv2
import numpy as np
import torch
import base64



####################################################################################

# GREEN_LIGHT = cv2.cvtColor(cv2.imread('ic/green.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
# RED_LIGHT = cv2.cvtColor(cv2.imread('ic/red.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)

GREEN_LIGHT =cv2.imread('core/ic/green.png')
RED_LIGHT = cv2.imread('core/ic/red.png')

TRAFFIC_LIGHT = {'green':GREEN_LIGHT, 'red':RED_LIGHT}
COLOR = {'black': (0,0,0), 'white': (255,255,255),'green': (0, 199, 0), 'yellow':(253, 251, 37), 'red':(255,0,0)}



device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# init license plate model
licensePlateModel = torch.hub.load('core/yolov5', 'custom', path='core/models/best_lp_4.pt', source='local').to(device)  # local repo
licensePlateModel.eval()

# init license plate model
model_detetect_number = torch.hub.load('core/yolov5', 'custom', path='core/models/detect_number_3.pt', source='local').to(device)  # local repo
model_detetect_number.eval()
###################################################################################
def EstimationSpeed(previous_point, current_point, previous_time, current_time):
        """
        kc: float(m)

        """
        px,py = previous_point
        qx,qy = current_point

        t = current_time - previous_time
        #kc = 2*math.asin(math.sqrt(math.sin((px-qx)/2)**2 +math.cos(px)* math.cos(qx)*(math.sin(py-qy)/2)**2))
        
        kc = math.sqrt((qx-px)**2+(qy-py)**2)
        v = kc/(t/3600)
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
def ChangePositionInList(lst, position1, position2):
    temp = lst[position1].copy()
    lst[position1] = lst[position2]
    lst[position2] = temp
    return lst

def applySummFunctin(img, axis = 1):
    res = np.sum(img, axis = axis) 
    return res

def splitLine(img):
    if (img.shape[0]/img.shape[1])<(1/3):
        return None

    s1 = applySummFunctin(img, axis =1)
    lst_max = np.where(s1== max(s1))[0]
    deadline = lst_max[int(len(lst_max)/2)]
    if deadline<(1/3)*len(s1) - deadline or deadline>(1/3)*len(s1) - deadline :
        deadline = int(len(s1)/2)    
    return deadline


def sort(bboxes, deadline):
    if deadline == None:
        return bboxes[np.argsort(bboxes[:,-2])]
    else:
        l1 = bboxes[bboxes[:,-1]<deadline]
        l2 = bboxes[bboxes[:,-1]>deadline]
        return np.concatenate((l1[np.argsort(l1[:,-2])], l2[np.argsort(l2[:,-2])]))


def get_license_plate(image ):

        bboxes = licensePlateModel(image).pandas().xyxy[0].to_numpy()
        if len(bboxes)==0:
            return None, 'Cant detect vehicle !'
        bboxes = bboxes[bboxes[:,5]==max(bboxes[:,5])][0]
        xmin, ymin, xmax, ymax = bboxes[0:4]
        img_license_plate = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        return img_license_plate, None

def recognize_plate(img_plate):
    img_plate =cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)#COLOR_BGR2GRAY
    bboxes = model_detetect_number(img_plate).pandas().xyxy[0].to_numpy()
    bboxes = np.append(bboxes, (bboxes[:,0]+(bboxes[:,2]-bboxes[:,0])/2).reshape(-1,1), axis = 1)
    bboxes =  np.append(bboxes, (bboxes[:,1]+(bboxes[:,3]-bboxes[:,1])/2).reshape(-1,1), axis = 1)
    deadline = splitLine(img_plate)
    return ''.join(sort(bboxes, deadline)[:,6])

def Full_Detect(image):
    img_plate,err= get_license_plate(image)
    if err != None:
        return 'can\'t detection'   
    return recognize_plate(img_plate)


def decodemIMG(blob_img):
    image_data = base64.b64decode(blob_img)
    np_array = np.frombuffer(image_data, np.uint8)
    return  cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

def encodemIMG(img):
    _, buffer = cv2.imencode('.jpg', img)
    return  base64.b64encode(buffer)


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
        # self.image_red_light = READ_LIGHT
        # self.image_green_light = GREEN_LIGHT
        self.img_light_traffic =TRAFFIC_LIGHT[light]
        self.time_each_loop = time_each_loop
        self.stop = False
    def run(self):
        while(not self.stop):
            time.sleep(self.time_each_loop)
            if self.light == 'red':
                self.light = 'green'
                self.img_light_traffic = TRAFFIC_LIGHT['green']
            else:
                self.light = 'red'
                self.img_light_traffic= TRAFFIC_LIGHT['red']

