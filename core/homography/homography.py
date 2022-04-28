import cv2 
import numpy as np

class Homography:
    def __init__(self,dst_size ,area_src):
        """
        dst_size: tuple (w,h) of mini map
        area_src: list point: [[tl],[tr],[bl],[br]] ex:[[682,60],[912,76],[424,413],[1217,467]]
        """
        self.w,self.h = dst_size
        area_src =np.array( area_src, dtype=np.float32)
        area_dst =  np.array([[0,0],[self.w,0],[0,self.h],[self.w,self.h]], dtype=np.float32)
        self.matrix_homography = cv2.getPerspectiveTransform(area_src, area_dst)
    def GetMatrix(self):
        return self.matrix_homography
    def Compute_homography(self, point):
        """
        point: type tuple , list(w,h)
        """
        np_point = np.append(np.array(point),1).reshape(3,1)
        homography_point = self.matrix_homography.dot(np_point).reshape((-1))
        homography_point = homography_point/homography_point[-1]
        return tuple(homography_point[:-1].astype(np.float32))
    def create_mini_map(self,src, bg_img = True):
        if bg_img:
            return cv2.warpPerspective(src, self.matrix_homography, (int(self.w),int(self.h)))
        return np.zeros((int(self.h),int(self.w),3), np.uint8)

# COLOR = {'black': (0,0,0), 'white': (255,255,255),'green': (0, 199, 0), 'yellow':(253, 251, 37), 'red':(255,0,0)}


# w,h = 300,400
# area_src = np.array([[682,60],[912,76],[424,413],[1217,467]], dtype=np.float32)
# area_dst =  np.array([[0,0],[w,0],[0,h],[w,h]], dtype=np.float32)
# src = cv2.imread('imgs/highway.png')
# # src = cv2.polylines(src, area_src.astype(np.int32).reshape(1,-1,2).transpose((3,2)), True, COLOR['red'], thickness=1)


# points = np.array([[750,250,1]]).reshape((3,1))

# H,_ = cv2.findHomography(area_src, area_dst,cv2.RANSAC,5.0))
# # matrix = cv2.getPerspectiveTransform(area_src, area_dst)
# # print(matrix)
# output = cv2.warpPerspective(src, H, (w,h))


# #############################################
# homography_p= H.dot(points).reshape((-1))
# homography_p = homography_p/homography_p[-1]
# print(set(homography_p[:-1].astype(int)) )
# src = cv2.circle(src, (750,250), 3, COLOR['yellow'], thickness=-2)
# output = cv2.circle(output, tuple(homography[:-1].astype(int)) , 3, COLOR['yellow'], thickness=-2)


# cv2.imshow("Original image", src)
# cv2.imshow('output', output)
# cv2.waitKey(0)