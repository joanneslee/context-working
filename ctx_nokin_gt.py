'''
6/9/22 1:00
Date	MTML_position_x	MTML_position_y	MTML_position_z	MTML_orientation_x	MTML_orientation_y	MTML_orientation_z	MTML_orientation_w	MTML_gripper_angle	MTMR_position_x	MTMR_position_y	MTMR_position_z	MTMR_orientation_x	MTMR_orientation_y	MTMR_orientation_z	MTMR_orientation_w	MTMR_gripper_angle	PSML_position_x	PSML_position_y	PSML_position_z	PSML_orientation_x	PSML_orientation_y	PSML_orientation_z	PSML_orientation_w	PSML_gripper_angle	PSMR_position_x	PSMR_position_y	PSMR_position_z	PSMR_orientation_x	PSMR_orientation_y	PSMR_orientation_z	PSMR_orientation_w	PSMR_gripper_angle
'''
import os, sys
import json
import pathlib
import math
from turtle import position
from xmlrpc.client import Boolean
from cv2 import KeyPoint, threshold
from matplotlib.colors import cnames
import numpy as np
from PIL import Image, ImageDraw, ImageColor,ImageFont
from scipy import interpolate
import time
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from shapely.geometry import LineString
#from dataclasses import dataclass
#from itertools import accumulate
from scipy import ndimage
from imantics import Polygons, Mask


global MAX_LEN
MAX_LEN = 200

global Gripperthreshold
Gripperthreshold = -0.8

mathematicaColors = {
    "blue":"#5E81B5",
    "orange":"#E09C24",
    "red":"#EA5536",
    "purple":"#A5609D",
    "green":"#8FB131",
    "blue" :"#5e9ec8",
    "olive":"#929600",
    "terracotta":"#C56E1A",
    "yellow":"#FEC000",
}
# list of colors for the annotations
colors =["#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]
# opacity of the annotation masks. Values range from (0 to 255) Type tuple
opacity = (180,)
# radius of keypoint
radius = 3

def main():     
    dir=os.getcwd()
    '''
    try:
        task=sys.argv[1]
        #print(task)
    except:
        print("Error: no task provided \nUsage: python draw_labels.py <task>")
        available_tasks = next(os.walk(os.path.join(dir, "images")))[1]
        print("Available task images: ", available_tasks)
        available_labels = next(os.walk(os.path.join(dir, "labels")))[1]
        print("Available task labels: ", available_labels)
        sys.exit()
    ''' 
    task = "Knot_Tying"
    I = Iterator(task)
    #I.DrawLabelsContextKin()
    I.DrawLabelsContext()
    quit();    
   
'''
JSONInterface deals with the particular JSON format of the annotations
It's set up now to read labels as we received them from Cogito

If the JSON annotations are in a different format, you can edit the getPolygons and getKeyPoints methods
'''
class JSONInterface:    
    def __init__(self, jsonLoc):
        self.json_location = jsonLoc    
        with open(self.json_location) as f:
            data = json.load(f)
            self.data = data
            self.meta = data['metadata']
            self.instances = data['instances']      
    '''
    Returns a list of polygons
    each polygon is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    '''
    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polygon"):                
                polygonSeries.append(instance["points"])   
                cn.append(instance["className"])
        return cn, polygonSeries
    '''
    Returns a list of PolyLines
    each polyline is a list of points ordered as [x1, y1, x2, y2, ... , xn, yn]
    '''
    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "point"):                
                keyPoints.append([instance['x'], instance['y']])   
                cn.append(instance["className"])
        return cn, keyPoints

    '''
    Returns a list of PolyLines
    each PolyLine is a list [x, y]
    '''
    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polyline"):                
                polylineSeries.append(instance["points"])    
                cn.append(instance["className"])
        return cn,polylineSeries

class NPYInterface2:
    def __init__(self):
        pass

    def getIntersection_cached(self, grasper_gt,grasper,threadSource):
        [grasper_gt,grasper] = grasper_gt,grasper
        [thread_gt,thread] = np.load(threadSource,allow_pickle=True)
        grasper[grasper>0.95]=1 #! instead of 0.97
        grasper[grasper<0.95]=0 #! instead of 0.97
        thread[thread>0.95]=1 
        thread[thread<0.95]=0 
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        #drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i,j) = self.isIntersecting(grasper,thread)
        #print(inter)
        return (y_center,x_center), (i,j), inter
    def getIntersectionFast(self, grasperSource,threadSource):
        [grasper_gt,grasper] = np.load(grasperSource,allow_pickle=True)
        [thread_gt,thread] = np.load(threadSource,allow_pickle=True)
        grasper[grasper>0.95]=1 #! instead of 0.97
        grasper[grasper<0.95]=0 #! instead of 0.97
        thread[thread>0.95]=1 
        thread[thread<0.95]=0 
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = (0,0)
        #drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i,j) = self.isIntersecting(grasper,thread)
        #print(inter)
        return (y_center,x_center), (i,j), inter

    def getIntersection(self, grasperSource,threadSource):
        [grasper_gt,grasper] = np.load(grasperSource,allow_pickle=True)
        [thread_gt,thread] = np.load(threadSource,allow_pickle=True)
        grasper[grasper>0.95]=1 #! instead of 0.97
        grasper[grasper<0.95]=0 #! instead of 0.97
        thread[thread>0.95]=1 
        thread[thread<0.95]=0 
        grasper = np.squeeze(grasper)
        thread = np.squeeze(thread)
        (x_center, y_center) = ndimage.center_of_mass(grasper)
        #drawObject = plt.Circle((y_center,x_center),radius=10,color='red', fill=True)
        inter, (i,j) = self.isIntersecting(grasper,thread)
        #print(inter)
        return (y_center,x_center), (i,j), inter

    def isIntersecting(self,maskA,maskB):
        rows = len(maskA)
        cols = len(maskA[0])
        for i in range(rows):
            for j in range(cols):
                if(maskA[i][j] == 1 and maskB[i][j] == 1):
                    return True, (i,j)
        return False, (0,0)

class NPYInterface:
    def __init__(self, npyLoc):
        self.grasper_loc = npyLoc
        self.thread_loc = npyLoc.replace("deeplab_grasper_v1","deeplab_thread_v1")
        self.needle_loc = npyLoc.replace("deeplab_grasper_v1","deeplab_needle_v1")
        self.ring_loc = npyLoc.replace("deeplab_grasper_v1","deeplab_ring_v1")
        #with open(self.npyarr_location) as f:
        self.grasperData = np.load(self.grasper_loc, allow_pickle=True)
        print("\t\t\t GrasperData",type(self.grasperData),self.grasperData.shape)
        self.g1 = self.grasperData[0]
        self.g2 = self.grasperData[1]
        print("\t\t\t l1",self.g1.shape,"l2",self.g2.shape)
        try:
            self.threadData = np.load(self.thread_loc, allow_pickle=True)
            print("\t\t\t ThreadData",self.threadData.shape)
            self.t1 = self.threadData[0]
            self.t2 = self.threadData[1]
            print("\t\t\t T1",self.t1.shape,"T2",self.t2.shape)
        except:
            pass
        try:
            self.needleData = np.load(self.needle_loc,allow_pickle=True) 
        except:
            pass

        try:
            self.ringData = np.load(self.ring_loc,allow_pickle=True)
        except:
            pass     
        self.instances = []

    def getPolygons(self):
        polygonSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polygon"):                
                polygonSeries.append(instance["points"])   
                cn.append(instance["className"])
        return cn, polygonSeries

    def getKeyPoints(self):
        keyPoints = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "point"):                
                keyPoints.append([instance['x'], instance['y']])   
                cn.append(instance["className"])
        return cn, keyPoints

    def getPolyLines(self):
        polylineSeries = list()
        cn = list()
        for instance in self.instances:            
            instance_ID = instance["classId"]
            instance_type = instance["type"]
            instance_probability = instance["probability"]
            instance_class = instance["className"]
            if(instance_type == "polyline"):                
                polylineSeries.append(instance["points"])    
                cn.append(instance["className"])
        return cn,polylineSeries

class MPInterface:
    def __init__(self,MPLoc):
        self.mp_loc = MPLoc
        self.transcript = []
        with open(self.mp_loc) as file:
            for line in file:
                #print(line.rstrip())
                self.transcript.append(line.rstrip())
    def getMP(self, index):
        #print("GetMP Matching",index)
        for i in range(1,len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if(int(l_s[1]) > index):
                return " ".join(l_s)

class ContextInterface:
    def __init__(self, ContextLoc,enable):
        if(enable):
            return
        self.c_loc = ContextLoc
        self.transcript = []
        try:
            with open(self.c_loc) as file:
                for line in file:
                    self.transcript.append(line.rstrip())
            self.empty = len(self.transcript) == 0
        except Exception as e:
            #print(e)
            pass

    def getContext(self,index):
        if(self.empty):
             return "Building Prediction"
        for i in range(1,len(self.transcript)):
            l_s = self.transcript[i].split(" ")
            if(int(l_s[0]) > index):
                #return " ".join(min(0,i-1))
                return self.transcript[max(0,i-1)]

        
class Iterator:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")
        self.labelsDir = os.path.join(self.CWD, task,"annotations")
        self.outputDir = os.path.join(self.CWD, task,"labeled_images")
        self.mpDir = os.path.join(self.CWD, task, "motion_primitives_combined")
        
        self.mpDir_R = os.path.join(self.CWD, task, "motion_primitives_R")
        
        self.mpDir_L = os.path.join(self.CWD, task, "motion_primitives_L")

        self.ContextDir = os.path.join(self.CWD,task,"transcriptions")

        self.kinDir = os.path.join(self.CWD,task,"kinematics")

        self.context_output = os.path.join(self.CWD,task,"vis_context_labels_gt")
        
        self.OS = "windows"

    """
    Iterator: getGripperAngle() 

    Pre: self.kinDir set to kinematics folder inside the <self.task> parent folder

    :return: touple of three columns from the kinematics data
    all_kin_data    =   [PSML_Gripper_Angle,PSMR_Gripper_Angle]
    all_kin_data_L  =   PSML_Gripper_Angle
    all_kin_data_R  =   PSMR_Gripper_Angle
    """ 
    def getGripperAngle(self):
        count = 0
        all_kin_data = {}
        all_kin_data_L = {}
        all_kin_data_R = {}
        for root, dirs, files in os.walk(self.kinDir):
            for file in files:
                if("99"  in file):
                    continue                
                #Suturing index 10 is PSML_gripper_angle
                #Suturing index 21 is PSMR_gripper_angle
                kin_file =  os.path.join(self.kinDir, file)                
                PSML_Gripper_Angle = []
                PSMR_Gripper_Angle = []
                with open(kin_file) as kin_data:
                    index=0
                    for line in kin_data:
                        if(index==0):
                            index+=1
                            continue                        
                        l_s = line.strip().split(",")
                        PSML_Gripper_Angle.append(l_s[10])
                        PSMR_Gripper_Angle.append(l_s[21])
                        index+=1      

                #print("\tKin_Length: ", kin_len)
                count=count+1
                all_kin_data[file] = [PSML_Gripper_Angle,PSMR_Gripper_Angle]
                all_kin_data_L[file]=PSML_Gripper_Angle
                all_kin_data_R[file]=PSMR_Gripper_Angle
        print("Obtaining Gripper Angle from Kinematic files...\t",count,self.task,"trials found!")
        return all_kin_data,all_kin_data_L,all_kin_data_R

    def getKinLines(self):
        count = 0
        all_kin_data = {}
        for root, dirs, files in os.walk(self.kinDir):
            for file in files:
                if("99"  in file):
                    continue
                #print(file)
                kin_file =  os.path.join(self.kinDir, file)

                kin_lines = []
                with open(kin_file) as kin_data:
                    for line in kin_data:
                        kin_lines.append(line.strip())

                i = 0 
                for line in kin_lines:
                    line_ = line.replace("\n","")
                    i=i+1
                kin_len = i;
                #print("\tKin_Length: ", kin_len)
                count=count+1
                all_kin_data[file] = kin_lines         

        print(count,"files processed!")
        return all_kin_data

    def imageToJSON(self, file):
        fileArr = file.split(".")
        return "".join(fileArr[:-1]) + ".json"

    def imageToNPY(self,file):
        fileArr = file.split(".")
        temp = "".join(fileArr[:-1]) + ".npy"
        return temp.replace(".npy","_gt_pred.npy")

    def getRBGA(self, hexColor):
        c = ImageColor.getcolor(hexColor, "RGB")        
        c = c + opacity
        return c

    def Centroid(self, points):
        length = len(points)
        x_arr = [];
        x_len = 0;
        y_arr = [];
        y_len = 0;
        for i in range(0,length):
            if( i %2==0):
                x_arr.append(points[i]);
                x_len=x_len+1;
            else:
                y_arr.append(points[i]);
                y_len = y_len+1;
        sum_x = np.sum(x_arr)
        sum_y = np.sum(y_arr)

        return sum_x/x_len, sum_y/y_len;
        
    def reorderPolyLines(self, polyLines):
        newPolyLines = [];
        endPoints = [];
        for p in polyLines:
            x0 = p[0]
            y0 = p[1]
            xn = p[-2]
            yn = p[-1]
            endPoints.append( [ [x0,xn],[y0,yn] ] ) # An,Bn       
        return newPolyLines

    ''' Iterator: DrawSingleImageKT(imageSource, labelSource, target, DEBUG=False)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    :param DEBUG:       boolean - enable console printout

    :return: nothing'''     
    def DrawSingleImageKT(self, imageSource, labelSource, target, DEBUG=False):
        
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); #! graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); #! None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        if self.OS == "windows":
            font = ImageFont.truetype("arial.ttf", 12)
        else: 
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")

        #! print(polyLines)
        #! return    
        img = Image.open(imageSource)
        #! draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   
        polyNames_TL = []
        polyLines_TL = []
        polyNames_TR = []
        polyLines_TR = []
        polyNames_BL = []
        polyLines_BL = []
        polyNames_BR = []
        polyLines_BR = []
        for i in range(len(polyLines)):
            if("Top Left" in polyLineNames[i]):
                polyNames_TL.append(polyLineNames[i])
                polyLines_TL.append(polyLines[i])
            elif("Top Right" in polyLineNames[i]):
                polyNames_TR.append(polyLineNames[i])
                polyLines_TR.append(polyLines[i])
            elif("Bottom Left" in polyLineNames[i]):
                polyNames_BL.append(polyLineNames[i])
                polyLines_BL.append(polyLines[i])
            elif("Bottom Right" in polyLineNames[i]):
                polyNames_BR.append(polyLineNames[i])
                polyLines_BR.append(polyLines[i])
        if(len(polyLines_TL)>0):                
            kp = [polyLines_TL[0][0], polyLines_TL[0][1]]             
            self.DrawThread(polyLines_TL, polyNames_TL, kp, draw, font)
        if(len(polyLines_TR)>0):                
            kp = [polyLines_TR[0][0], polyLines_TR[0][1]]             
            self.DrawThread(polyLines_TR, polyNames_TR, kp, draw, font)
        if(len(polyLines_BL)>0):                
            kp = [polyLines_BL[0][0], polyLines_BL[0][1]]             
            self.DrawThread(polyLines_BL, polyNames_BL, kp, draw, font)
        if(len(polyLines_BR)>0):                
            kp = [polyLines_BR[0][0], polyLines_BR[0][1]]             
            self.DrawThread(polyLines_BR, polyNames_BR, kp, draw, font)

        img.save(target)
        return 
    
    ''' Iterator: CalcDistancesSingleThread(self,LGX,LGY,RGX,RGY,ThreadX,ThreadY)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    MPI, CtxI,CtxI_Pred,
    :param DEBUG:       boolean - enable console printout

    :do: Draw a single image containing
        - ground truth context labels
        - predicted context labels, 
        - annotation objects (gripper mask, thread mask)

        using the video frame as background
    LG_Info, RG_Info, LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextKT(imageSource,labelSource,outputDest, MPI, CtxI,CtxI_Pred)
    #! used in DrawSingleImageContextKT  
    :return: '''        
    def CalcDistancesSingleThread(self,LGX,LGY,RGX,RGY,ThreadX,ThreadY):
        LG_Points = []
        RG_Points = []
        Thread_Points = []
        for i in range(len(LGX)):
            LG_Points.append(  (LGX[i],LGY[i]) )
        for i in range(len(RGX)):
            RG_Points.append(  (RGX[i], RGY[i])   )
        for i in range(len(ThreadX)):
            Thread_Points.append(  (ThreadX[i], ThreadY[i])   )

        LG = Polygon(LG_Points)
        RG = Polygon(RG_Points)
        Thread = LineString(Thread_Points)

        LG_Info = []
        try:
            LG_Info = [ LG.distance(Thread), LG.intersects(Thread) ]
        except Exception as e:
            #print(e)
            LG_Info = [e,""]

        RG_Info = []
        try:
            RG_Info = [ RG.distance(Thread), RG.intersects(Thread) ]
        except Exception as e:
            #print(e)
            RG_Info = [e,""]
        
        return LG_Info,RG_Info

    def CalcDistances(self, LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY):
        LG_Points = []
        RG_Points = []
        N_Points = []
        RingPoints = [[]]

        for i in range(len(LGX)):
            LG_Points.append(  (LGX[i],LGY[i]) )
        for i in range(len(RGX)):
            RG_Points.append(  (RGX[i], RGY[i])   )
        for i in range(len(NX)):
            N_Points.append(    (NX[i],NY[i]))
        

        for j in range(len(RingsX)):
            RingPoints.append([])
            for i in range(len(RingsX[j])):
                RingPoints[j].append(   (RingsX[j][i],RingsY[j][i])  )



        LG = Polygon(LG_Points)
        RG = Polygon(RG_Points)
        Needle = Polygon(N_Points)

        RingsArr = []
        for j in range(len(RingsX)):
            RingsArr.append( Polygon(RingPoints[j]))

        LG_Info = []
        try:
            LG_Info = [ LG.distance(Needle), LG.intersects(Needle) ]
        except Exception as e:
            #print(e)
            LG_Info = [e,""]
        RG_Info = []
        try:
            RG_Info = [ RG.distance(Needle), RG.intersects(Needle) ]
        except Exception as e:
            #print(e)
            RG_Info = [e,""]
        #N_Info = ["None","None"]

        intersections = False
        interIDX = -1
        d = []
        for j in range(len(RingsX)):
            try:
                if( Needle.intersects(RingsArr[j])):
                    intersections = True
                    interIDX = j
                d.append(Needle.distance(RingsArr[j]))
            except:
                pass
        
            

        if(intersections):
            N_Info = ["Intersecting Ring_"+str(interIDX+4),";"]
        else:
            #N_Info = ["No intersections", " ".join([str(s) for s in d])]
            index_min = np.argmin(d)
            N_Info = ["", "Needle is d="+str(d[index_min])+" to Ring_"+str(index_min+4) ]
        '''

        try:
            # I have 4 rings
            # let's see if there are any intersections:
            d = []
            intersections = False
            for j in range(len(RingsX)):
                if( Needle.intersects(RingsArr[j])):
                    Ints = True
                d.append(Needle.distance(RingsArr[j]))

            index_min = np.argmin(d)
            N_Info[0] = "No Inters"
            N_Info[1] = " d="+str(d[index_min])+" to Ring_"+str(index_min+4)

            if(intersections):
                for k in range(len(RingsX)):
                    if( Needle.intersects(RingsArr[k])):
                        N_Info[0] = "Inter Ring_"+str(k+4)
                        N_Info[1] = " d=0 to Ring_"+str(k+4)
                           
        except Exception as e:
            print(e)
        '''
        distances = d
        return LG_Info, RG_Info, N_Info, (interIDX+4), distances
        #return (LG.distance(Needle), LG.intersects(Needle)), (RG.distance(Needle), RG.intersects(Needle)), "rings"
   
    def OrganizePoints(self, polygons,polyNames):
        LGX = []
        LGY = []
        RGX = []
        RGY = []
        NX = []
        NY = []
        RingsX = [[],[],[],[]]
        RingsY = [[],[],[],[]]
        for i in range(len(polyNames)):
            if("Left Grasper" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    LGX.append(polygons[i][k])
                    LGY.append(polygons[i][k+1])
            elif("Right Grasper" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RGX.append(polygons[i][k])
                    RGY.append(polygons[i][k+1])
            elif("Needle Mask" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    NX.append(polygons[i][k])
                    NY.append(polygons[i][k+1])
            elif("Ring_4" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[0].append(polygons[i][k])
                    RingsY[0].append(polygons[i][k+1])            
            elif("Ring_5" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[1].append(polygons[i][k])
                    RingsY[1].append(polygons[i][k+1])
            elif("Ring_6" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[2].append(polygons[i][k])
                    RingsY[2].append(polygons[i][k+1])
            elif("Ring_7" in polyNames[i]):
                for k in range(0,len(polygons[i]),2):
                    RingsX[3].append(polygons[i][k])
                    RingsY[3].append(polygons[i][k+1])
            else:
                print("Unknown Polygon Class",polyNames[i])
        return LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY
    
    def CrossProduct(self,A):     
        # Stores coefficient of X
        # direction of vector A[1]A[0]
        X1 = (A[1][0] - A[0][0])
    
        # Stores coefficient of Y
        # direction of vector A[1]A[0]
        Y1 = (A[1][1] - A[0][1])
    
        # Stores coefficient of X
        # direction of vector A[2]A[0]
        X2 = (A[2][0] - A[0][0])
    
        # Stores coefficient of Y
        # direction of vector A[2]A[0]
        Y2 = (A[2][1] - A[0][1])
    
        # Return cross product
        return (X1 * Y2 - Y1 * X2)
 
    # Function to check if the polygon is
    # convex polygon or not
    def isConvex(self,points):
        
        # Stores count of
        # edges in polygon
        N = len(points)
    
        # Stores direction of cross product
        # of previous traversed edges
        prev = 0
    
        # Stores direction of cross product
        # of current traversed edges
        curr = 0
    
        # Traverse the array
        for i in range(N):
            
            # Stores three adjacent edges
            # of the polygon
            temp = [points[i], points[(i + 1) % N],
                            points[(i + 2) % N]]
    
            # Update curr
            curr = self.CrossProduct(temp)
    
            # If curr is not equal to 0
            if (curr != 0):
                
                # If direction of cross product of
                # all adjacent edges are not same
                if (curr * prev < 0):
                    return False
                else:
                    
                    # Update curr
                    prev = curr
    
        return True

    def isIntersecting(self,maskA):
        rows = len(maskA)
        cols = len(maskA[0])
        for i in range(rows):
            for j in range(cols):
                if(maskA[i][j] == 1 and maskB[i][j] == 1):
                    return True, (i,j)
        return False, (0,0)

    ''' Iterator: DrawSingleImageContextKT(imageSource,imageSource, labelSource, target, MPI, CtxI,CtxI_Pred, DEBUG=False)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    MPI, CtxI,CtxI_Pred,
    :param DEBUG:       boolean - enable console printout

    :do: Draw a single image containing
        - ground truth context labels
        - predicted context labels, 
        - annotation objects (gripper mask, thread mask)

        using the video frame as background
    LG_Info, RG_Info, LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextKT(imageSource,labelSource,outputDest, MPI, CtxI,CtxI_Pred)

    :return: '''     
    #!LG_Info, RG_Info, LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextKT(imageSource,labelSource,outputDest, MPI, CtxI,CtxI_Pred)
    def DrawSingleImageContextKT(self, LgrasperMask,RgrasperMask, ThreadSource, target,CtxI,CtxI_Pred, DEBUG=False):       
        #J = JSONInterface(labelSource)
        Rgrasper = NPYInterface2()
        try:
            (x_p_R,y_p_R), (i_R,j_R), interR = Rgrasper.getIntersectionFast(RgrasperMask, ThreadSource) # (center), (intersection), boolean
        #print((x_p,y_p), inter)
        #Lgrasper = NPYInterface2()
            (x_p_L,y_p_L), (i_L,j_L), interL = Rgrasper.getIntersectionFast(LgrasperMask, ThreadSource) # (center), (intersection), boolean
        except Exception:
            interR = False
            interL = False


        #def isIntersecting(self,maskA,maskB):
        #rows = len(maskA)
        #cols = len(maskA[0])
        #for i in range(rows):
        #    for j in range(cols):
        #        if(maskA[i][j] == 1 and maskB[i][j] == 1):
        #            return True, (i,j)
        #return False, (0,0)


        #for i in range(0,len(x_p_L))

        #[grasper_gt,grasper] = np.load(RgrasperMask,allow_pickle=True)
        #grasper[grasper>0.95]=1 #! instead of 0.97
        #grasper[grasper<0.95]=0 #! instead of 0.97
        #grasper = np.squeeze(grasper)
        #polygons = Mask(grasper).polygons()
        #polygons.

        #print(polygons.points)
        #print(polygons.segmentation)

        LG_inter_t = False
        RG_inter_t = False
        LG_inter_t = interL
        RG_inter_t = interR
        #if(inter):
        #    if(i>x_p):
        #        RG_inter_t = True
        #    else:
        #        LG_inter_t=True  
        
        LG_Message = ""
        RG_Message = ""
        LG_Thread_Message = ""
        RG_Thread_Message = ""
        N_Message = ""
        Lens_Message = ""
        #! if N internsection is 3, there is no intersection
        #LG_Thread_Info_Top, RG_Thread_Info_Top = self.CalcDistancesSingleThread(LGX, LGY, RGX, RGY, SingleThreadXTop, SingleThreadYTop)
        #LG_Thread_Info_Bottom, RG_Thread_Info_Bottom = self.CalcDistancesSingleThread(LGX, LGY, RGX, RGY, SingleThreadXBottom, SingleThreadYBottom)
        LG_Thread_Info_Top, RG_Thread_Info_Top = [1,LG_inter_t],[1,RG_inter_t]
        LG_Thread_Info_Bottom, RG_Thread_Info_Bottom = [1,LG_inter_t],[1,RG_inter_t]
        if(not LG_Thread_Info_Top[1] and LG_Thread_Info_Bottom[1] and not RG_Thread_Info_Top[1] and not RG_Thread_Info_Bottom[1]):
            print("stop here")
        try:
            if(len(LG_Thread_Info_Top) == 0 or len(LG_Thread_Info_Bottom) == 0):
                LG_Message = "No Annotation for Left Grasper"
            else:
                LG_Message =    "L To Top Thread:   " + '{0:.2f}'.format(LG_Thread_Info_Top[0])  + " Inters:" + str(LG_Thread_Info_Top[1])
                LG_Message += "\nL To Bottom Thread:" + '{0:.2f}'.format(LG_Thread_Info_Bottom[0])  + " Inters:" + str(LG_Thread_Info_Bottom[1])

            if(len(RG_Thread_Info_Top) == 0 or len(RG_Thread_Info_Bottom) == 0):
                RG_Message = "No Annotation for Right Grasper"
            else:
                RG_Message =  "\nR To Top Thread:   " + '{0:.2f}'.format(RG_Thread_Info_Top[0])  + " Inters:" + str(RG_Thread_Info_Top[1])
                RG_Message += "\nR To Bottom Thread:" + '{0:.2f}'.format(RG_Thread_Info_Bottom[0])  + " Inters:" + str(RG_Thread_Info_Bottom[1])            
        except Exception as e: 
            print(e)        
        return LG_Thread_Info_Top,RG_Thread_Info_Top, LG_Thread_Info_Bottom,RG_Thread_Info_Bottom #! add LG Bottom, RG Top

    def DrawSingleImageContextNP(self, grasperMask,threadMask,needleMask,ringMask,outputDest, CtxI,CtxI_Pred):
        #J = JSONInterface(labelSource)
        prednpyG_T = NPYInterface2()
        prednpyG_N = NPYInterface2()
        prednpyG_R = NPYInterface2()
        prednpyR_N = NPYInterface2()
        [grasper_gt,grasper] = np.load(grasperMask,allow_pickle=True)
        (x_p_T,y_p_T), (i_T,j_T), inter_T = prednpyG_T.getIntersection_cached(grasper_gt,grasper, threadMask) # (center), (intersection), boolean
        (x_p_N,y_p_N), (i_N,j_N), inter_N = prednpyG_N.getIntersection_cached(grasper_gt,grasper, needleMask) 
        (x_p_R,y_p_R), (i_R,j_R), inter_R = prednpyG_R.getIntersection_cached(grasper_gt,grasper, ringMask) 
        (x_p_R_N,y_p_R_N), (i_R_N,j_R_N), inter_R_N = prednpyR_N.getIntersection(needleMask, ringMask) 
        #print((x_p_T,y_p_T), inter_T)
        polygons = Mask(grasper).polygons()

        print(polygons.points)
        print(polygons.segmentation)


        LG_inter_T = False
        RG_inter_T = False
        if(inter_T):
            if(i_T>x_p_T):
                RG_inter_T = True
            else:
                LG_inter_T=True  

        LG_inter_N = False
        RG_inter_N = False
        if(inter_N):
            if(i_N>x_p_N):
                RG_inter_N = True
            else:
                LG_inter_N=True  
        
        LG_inter_R = False
        RG_inter_R = False
        if(inter_R):
            if(i_R>x_p_R):
                RG_inter_R = True
            else:
                LG_inter_R=True          
        
        '''
        LG_Message = ""
        RG_Message = ""
        LG_Thread_Message = ""
        RG_Thread_Message = ""
        N_Message = ""
        Lens_Message = ""
        '''

        #SingleThreadX, SingleThreadY = "_", "_"
        #LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY = self.OrganizePoints(polygons,polyNames)
        #LG_Info, RG_Info, N_Info, N_Intersection, Needle_Ring_Distances = self.CalcDistances(LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY) 
        
        #LG_Thread_Info = [1,LG_inter_T]
        #RG_Thread_Info = [1,RG_inter_T]
        #if(not LG_Thread_Info_Top[1] and LG_Thread_Info_Bottom[1] and not RG_Thread_Info_Top[1] and not RG_Thread_Info_Bottom[1]):
        #    print("stop here")
        return LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,LG_inter_R,RG_inter_R,inter_R_N
        '''
        try:
            if(len(LGX) == 0 or len(LGY) == 0):
                LG_Message = "No Annotation for Left Grasper"
            else:
                LG_Message = "L To Needle:" + '{0:.2f}'.format(LG_Info[0])  + " Inters:" + str(LG_Info[1])
            if(len(RGX) == 0 or len(RGY) == 0):
                RG_Message = "No Annotation for Right Grasper"
            else:
                RG_Message = "R To Needle:" + '{0:.2f}'.format(RG_Info[0])  + " Inters:" + str(RG_Info[1])
            
            if(len(NX) == 0 or len(NY) == 0):
                N_Message = "No annotation for Needle Or Rings"
            else:
                #N_Message = '{0:.2f}'.format(N_Info[0]) + " : " + '{0:.2f}'.format(N_Info[1]) 
                N_Message = N_Info[0] + " : " + N_Info[1]

            if(len(LGX) == 0 or len(LGY) == 0 or len(SingleThreadX) == 0):
                LG_Thread_Message = "No Annotation for L_G_Thread"
            else:
                LG_Thread_Message = "L To Thread:" + '{0:.2f}'.format(LG_Thread_Info[0])  + " Inters:" + str(LG_Thread_Info[1])

            if(len(RGX) == 0 or len(RGY) == 0 or len(SingleThreadX) == 0):
                RG_Thread_Message = "No Annotation for R_G_Thread"
            else:
                RG_Thread_Message = "R To Thread:" + '{0:.2f}'.format(RG_Thread_Info[0])  + " Inters:" + str(RG_Thread_Info[1])           
        except Exception as e: 
            print(e)  
                  
        return LG_Info, RG_Info, N_Intersection, Ring_Intersection,LG_Thread_Info, RG_Thread_Info
        '''
    
    def DrawSingleImageContextS(self, grasperMask,threadMask,needleMask,outputDest, CtxI,CtxI_Pred):
        prednpyG_T = NPYInterface2()
        prednpyG_N = NPYInterface2()
        (x_p_T,y_p_T), (i_T,j_T), inter_T = prednpyG_T.getIntersection(grasperMask, threadMask) # (center), (intersection), boolean
        (x_p_N,y_p_N), (i_N,j_N), inter_N = prednpyG_N.getIntersection(grasperMask, needleMask) 
        #print((x_p_T,y_p_T), inter_T)
        LG_inter_T = False
        RG_inter_T = False
        if(inter_T):
            if(i_T>x_p_T):
                RG_inter_T = True
            else:
                LG_inter_T=True  
        LG_inter_N = False
        RG_inter_N = False
        if(inter_N):
            if(i_N>x_p_N):
                RG_inter_N = True
            else:
                LG_inter_N=True  
        return LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N
    
    ''' Iterator: DrawSingleImageContext(imageSource,imageSource, labelSource, target, MPI, CtxI,CtxI_Pred, DEBUG=False)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    MPI, CtxI,CtxI_Pred,
    :param DEBUG:       boolean - enable console printout

    :do: Draw a single image containing
        - ground truth context labels
        - predicted context labels, 
        - annotation objects (gripper mask, thread mask)

        using the video frame as background
    LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContext(imageSource,labelSource,outputDest, MPI, CtxI,CtxI_Pred)

    :return:''' 
    def DrawSingleImageContext(self, imageSource, labelSource,target, CtxI,CtxI_Pred, DEBUG=False):       
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();
        SingleThreadX = []
        SingleThreadY = []

        if self.OS == "windows":
            font = ImageFont.truetype("arial.ttf", 12)
        else: 
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")

        #font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")
        
        img = Image.open(imageSource)
        IDX = int(imageSource.split("_")[-1].split(".")[0])
        draw = ImageDraw.Draw(img, "RGBA")   

        self.DrawPolygons(polygons,polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)

        if("Needle End" not in kpNames):
            #print("No needle for", imageSource)
            pass
        else: 
            needleEnd = KeyPoint[0]
            for i in range(len(KeyPoint)):
                if("Needle End" in kpNames[i]):
                    needleEnd = KeyPoint[i]
            if(len(polyLines) !=0):     
                # what is the most useful object we can get out of self.DrawThread that would help us determine intersections with the thread? 
                SingleThreadX, SingleThreadY = self.DrawThread(polyLines, polyLineNames, needleEnd, draw, font)
        
        
        LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY = self.OrganizePoints(polygons,polyNames)
        LG_Message = ""
        RG_Message = ""
        LG_Thread_Message = ""
        RG_Thread_Message = ""
        

        N_Message = ""
        Lens_Message = ""
        # if N internsection is 3, there is no intersection
        LG_Info, RG_Info, N_Info, N_Intersection, Needle_Ring_Distances = self.CalcDistances(LGX,LGY,RGX,RGY,NX,NY,RingsX,RingsY) 
        LG_Thread_Info, RG_Thread_Info = self.CalcDistancesSingleThread(LGX, LGY, RGX, RGY, SingleThreadX, SingleThreadY)
        
        try:
            if(len(LGX) == 0 or len(LGY) == 0):
                LG_Message = "No Annotation for Left Grasper"
            else:
                LG_Message = "L To Needle:" + '{0:.2f}'.format(LG_Info[0])  + " Inters:" + str(LG_Info[1])
            if(len(RGX) == 0 or len(RGY) == 0):
                RG_Message = "No Annotation for Right Grasper"
            else:
                RG_Message = "R To Needle:" + '{0:.2f}'.format(RG_Info[0])  + " Inters:" + str(RG_Info[1])
            
            if(len(NX) == 0 or len(NY) == 0):
                N_Message = "No annotation for Needle Or Rings"
            else:
                #N_Message = '{0:.2f}'.format(N_Info[0]) + " : " + '{0:.2f}'.format(N_Info[1]) 
                N_Message = N_Info[0] + " : " + N_Info[1]

            if(len(LGX) == 0 or len(LGY) == 0 or len(SingleThreadX) == 0):
                LG_Thread_Message = "No Annotation for L_G_Thread"
            else:
                LG_Thread_Message = "L To Thread:" + '{0:.2f}'.format(LG_Thread_Info[0])  + " Inters:" + str(LG_Thread_Info[1])

            if(len(RGX) == 0 or len(RGY) == 0 or len(SingleThreadX) == 0):
                RG_Thread_Message = "No Annotation for R_G_Thread"
            else:
                RG_Thread_Message = "R To Thread:" + '{0:.2f}'.format(RG_Thread_Info[0])  + " Inters:" + str(RG_Thread_Info[1])

        except Exception as e: 
            print(e)

        Lens_Message = "L:" + str(len(LGX) - len(LGY)) + " R:" + str( len(RGX) - len(RGY)) + " N:" + str(len(NX)-len(NY)) 
        # use Lens_Message inside first argument of DrawTextArr        
        try:
            self.DrawTextArr([CtxI_Pred.getContext(IDX)+"-predicted",CtxI.getContext(IDX),LG_Message, RG_Message,LG_Thread_Message,RG_Thread_Message,N_Message ], draw, font)
        except Exception as e: 
            self.DrawTextArr(["Shape Exception" ], draw, font)        
        img.save(target) # to save        
        #F.L_G_Dist = LG_Info[0]
        
        return LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info

    def DrawSingleImage(self, imageSource, labelSource, target, MPI, CtxI, DEBUG=False):
        J = JSONInterface(labelSource)
        polyNames , polygons = J.getPolygons(); # graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); # None in KT,
        polyLineNames, polyLines = J.getPolyLines();

        
        if self.OS == "windows":
            font = ImageFont.truetype("arial.ttf", 12)
        else: 
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")

        #print(polyLines)
        #return    
        img = Image.open(imageSource)
        IDX = int(imageSource.split("_")[-1].split(".")[0])
        #draw = ImageDraw.Draw(img)
        draw = ImageDraw.Draw(img, "RGBA")   

        self.DrawPolygons(polygons,polyNames,draw,font)
        self.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)
        #self.DrawTextTopCorner(MPI.getMP(IDX),draw,font)
        self.DrawTextTopCorner(CtxI.getContext(IDX), draw, font)

        distances = self.CalcDistances(polygons,polyNames,polyLines,polyLineNames)

        if("Needle End" not in kpNames):
            #print("No needle for", imageSource)
            pass
        else: 
            needleEnd = KeyPoint[0]
            for i in range(len(KeyPoint)):
                if("Needle End" in kpNames[i]):
                    needleEnd = KeyPoint[i]
            if(len(polyLines) !=0):     
                self.DrawThread(polyLines, polyLineNames, needleEnd, draw, font)

        img.save(target) # to save
        return

    def RenderThread_Arr(self, thread_X, thread_Y, draw, font):        
        kk=0   
        t_min = min(len(thread_X),len(thread_Y))
        thread_X = thread_X[0:t_min]
        thread_Y = thread_Y[0:t_min]       
        distance = np.cumsum(np.sqrt( np.ediff1d(thread_X, to_begin=0)**2 + np.ediff1d(thread_Y, to_begin=0)**2 ))
        distance = distance/distance[-1]        
        fx, fy = interp1d( distance, thread_X ), interp1d( distance, thread_Y )                
        alpha = np.linspace(0, 1, 10)
        alpha_context = np.linspace(0, 1, 100)
        x_regular, y_regular = fx(alpha), fy(alpha)
        x_detailed, y_detailed = fx(alpha_context), fy(alpha_context)
        for jj in range(0,len(thread_X)):           
            draw.line((thread_X[kk],
                    thread_Y[kk],
                    thread_X[kk+1],
                    thread_Y[kk+1]), fill=(255, 0, 0, 127), width=2) 
            kk+=1
            if(kk>=len(thread_X)-1): break
        for i in range(len(x_regular)): # draws each KeyPoint
            x = x_regular[i]
            y = y_regular[i]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            #c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList, fill=(0,0,0))
        return x_detailed, y_detailed
    
    def dist(self, A, B):
        return math.sqrt( (A[0] - B[0] )**2+( A[1]-B[1])**2 )

    def distNeedle(self, ax, ay,needleEnd ):
        return math.sqrt( (ax - needleEnd[0] )**2+( ay - needleEnd[1] )**2 )

    #! PutTheadInOrder(polyLines arr,needleEnd as keypoint)
    def PutTheadInOrder(self,polyLines_M,kp):
        thread_X = []
        thread_Y = []  
        polyLines = polyLines_M.copy()
        while(len(polyLines) > 0):    
            index_of_closest_thread_arr = 0; 
            position_of_closest = 0; #!  0 for "cannonical" - same position , 1 for end
            thread_idx_0_distances = []
            thread_idx_1_distances = [] 

            for i in range(len(polyLines)):
                l = len(polyLines[i])
                fx = polyLines[i][0]
                fy = polyLines[i][1]
                lx = polyLines[i][l-2]
                ly = polyLines[i][l-1]
                thread_idx_0_distances.append(self.distNeedle(fx,fy,kp))
                thread_idx_1_distances.append(self.distNeedle(lx,ly,kp))       

            min_0 = min(thread_idx_0_distances) #! min of all distances between last end
            min_1 = min(thread_idx_1_distances)

            idx_0 = thread_idx_0_distances.index(min_0)
            idx_1 = thread_idx_1_distances.index(min_1)
            if(min_0 < min_1):
                position_of_closest = 0
                index_of_closest_thread_arr = idx_0
            else:
                position_of_closest = 1
                index_of_closest_thread_arr = idx_1
            
            shortest = polyLines[index_of_closest_thread_arr]
            if(position_of_closest ==0):
                for j in range(0,len(shortest),2):
                    thread_X.append(shortest[j])
                    thread_Y.append(shortest[j+1])
                    kp = [shortest[j],shortest[j+1]]
            else:
                ll = len(shortest)
                for j in range(0,len(shortest),2):
                    thread_X.append(shortest[ll-j-2])
                    thread_Y.append(shortest[ll-j-1])
                    kp = [shortest[ll-j-2],shortest[ll-j-1]]


            
            del polyLines[index_of_closest_thread_arr]



        return thread_X, thread_Y


    def DrawThreadKT(self,polyLines, polyLineNames, needleEnd, draw, font):  
        thread_X = []
        thread_Y = []
        if(len(polyLines)<2):
            for i in range(len(polyLines)):
                l = len(polyLines)
                for j in range(0,len(polyLines[i]),2):
                    thread_X.append(polyLines[i][j])
                    thread_Y.append(polyLines[i][j+1])
        else:
            thread_X, thread_Y = self.PutTheadInOrder(polyLines,needleEnd)
      

        '''
        for i in range(len(polyLines)):
            l = len(polyLines)
            c = self.getRBGA(colors[-i])
            k=0
            for j in range(0,len(polyLines[l-i-1]),2):
                thread_X.append(polyLines[l-i-1][j])
                thread_Y.append(polyLines[l-i-1][j+1])
        '''  
        for i in range(len(polyLines)):
            c = self.getRBGA(colors[-i])
            k=0          

            for j in range(0,len(polyLines[i])):
                #if(k+3>=len(polyLines[i])):
                #    
                draw.line(( polyLines[i][k],
                            polyLines[i][k+1],
                            polyLines[i][k+2],
                            polyLines[i][k+3]), fill=c, width=9) 
                x = polyLines[i][k]
                y = polyLines[i][k+1]          
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                x = polyLines[i][k+2]
                y = polyLines[i][k+3]     
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                k+=2
                if(k>=len(polyLines[i])-2): break
            draw.text((polyLines[i][0],polyLines[i][1]),polyLineNames[i]+str(i),(255,255,255),font=font)   

        ThreadX, ThreadY = self.RenderThread_Arr(thread_X, thread_Y, draw, font)
        return ThreadX, ThreadY  

    def DrawThread(self,polyLines, polyLineNames, needleEnd, draw, font):  
        thread_X = []
        thread_Y = []
        if(len(polyLines)<2):
            for i in range(len(polyLines)):
                l = len(polyLines)
                for j in range(0,len(polyLines[i]),2):
                    thread_X.append(polyLines[i][j])
                    thread_Y.append(polyLines[i][j+1])
        else:
            thread_X, thread_Y = self.PutTheadInOrder(polyLines,needleEnd)
      

        '''
        for i in range(len(polyLines)):
            l = len(polyLines)
            c = self.getRBGA(colors[-i])
            k=0
            for j in range(0,len(polyLines[l-i-1]),2):
                thread_X.append(polyLines[l-i-1][j])
                thread_Y.append(polyLines[l-i-1][j+1])
        '''  
        for i in range(len(polyLines)):
            c = self.getRBGA(colors[-i])
            k=0          

            for j in range(0,len(polyLines[i])):
                #if(k+3>=len(polyLines[i])):
                #    
                draw.line(( polyLines[i][k],
                            polyLines[i][k+1],
                            polyLines[i][k+2],
                            polyLines[i][k+3]), fill=c, width=9) 
                x = polyLines[i][k]
                y = polyLines[i][k+1]          
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                x = polyLines[i][k+2]
                y = polyLines[i][k+3]     
                leftUpPoint = (x-2, y-2)
                rightDownPoint = (x+2, y+2)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.rectangle(twoPointList,fill=(0, 255, 0, 255))
                k+=2
                if(k>=len(polyLines[i])-2): break
            draw.text((polyLines[i][0],polyLines[i][1]),polyLineNames[i]+str(i),(255,255,255),font=font)   

        ThreadX, ThreadY = self.RenderThread_Arr(thread_X, thread_Y, draw, font)
        return ThreadX, ThreadY
            
    def DrawKeyPoints(self,KeyPoint, kpNames, polygons,draw,font):
        for i in range(len(KeyPoint)): # draws each KeyPoint
            x = KeyPoint[i][0]
            y = KeyPoint[i][1]            
            leftUpPoint = (x-radius, y-radius)
            rightDownPoint = (x+radius, y+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList, fill=c)
            draw.text((x-radius*2, y-radius),kpNames[i]+str(i),(255,255,255),font=font)

    def DrawTextTopCorner(self,MPI_str,draw,font):
        if(MPI_str is None): 
            return           
        x = 50
        y = 50
        leftUpPoint = (x-radius, y-radius)
        rightDownPoint = (x+radius, y+radius)
        twoPointList = [leftUpPoint, rightDownPoint]
        #c = self.getRBGA(colors[(len(polygons))])
        #draw.ellipse(twoPointList, fill=c)
        draw.text((x-radius*2, y-radius),MPI_str,(255,255,255),font=font)

    def DrawTextArr(self,strArr,draw,font):
        if(len(strArr) == 0 or strArr[0] is None): 
            return
        offset = 1
        for s in strArr:
            x = 10
            y = 15 * offset; 
            draw.text( (x,y),s,(255,255,255),font=font);
            offset+=1

    def DrawPolygons(self, polygons,polyNames,draw,font):
        for i in range(len(polygons)):
            #if("Ring" in polyNames[i]):
            c = self.getRBGA(colors[i])
                #print("Poly1:",polygons[i])
            draw.polygon(polygons[i], fill=c) #,outline='#EA5536')     
                ########## CENTER POINT
            x_c, y_c = self.Centroid(polygons[i])          
            leftUpPoint = (x_c-radius, y_c-radius)
            rightDownPoint = (x_c+radius, y_c+radius)
            twoPointList = [leftUpPoint, rightDownPoint]
            c = self.getRBGA(colors[i+(len(polygons))])
            draw.ellipse(twoPointList,outline=1, fill=c)            
                # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)

    '''Loops through self.imagesDir '''    
    def isGripperClosed(self, frameNumber, L_Gripper_Angle,R_Gripper_Angle):
        global Gripperthreshold
        Gripperthreshold = -0.8
        '''
        max_angle_L = max(L_Gripper_Angle)
        min_angle_L = min(L_Gripper_Angle)
        max_angle_R = max(R_Gripper_Angle)
        min_angle_R = min(R_Gripper_Angle)
        '''
        L_Gripping = False
        R_Gripping = False

        if(frameNumber<len(L_Gripper_Angle) and frameNumber < len(R_Gripper_Angle)):
            L_Gripping = float(L_Gripper_Angle[frameNumber]) < Gripperthreshold
            R_Gripping = float(R_Gripper_Angle[frameNumber]) < Gripperthreshold

        return L_Gripping,R_Gripping

    """ Iterator: DrawLabelsContextKin() 
    Pre:self.kinDir set to path to kinematics folder
    :return: describe what it returns """ 
    def DrawLabelsContextKin(self):
        count = 0
        all_kin_data,all_kin_data_L,all_kin_data_R = self.getGripperAngle(); #! gets gripper angle kinematics columns as lists - from .../kinematics/ folder

        #! for each trial for which there are kinematics files with gripper angle - see Iterator.getGripperAngle()
        for trial in all_kin_data:
            # used as sample ID: Knot_Tying_S02_T02
            task_subject_trial = trial.replace(".csv","")

            #! Gripper Angles L+R, L, and R
            trialKin = all_kin_data[trial] 
            trialKin_L = all_kin_data_L[trial]
            trialKin_R = all_kin_data_R[trial]

            trialImageDir = os.path.join(self.imagesDir,task_subject_trial)
            ctxOutput = os.path.join(self.context_output,task_subject_trial+".txt")

            #if(os.path.exists(ctxOutput)):
            #    print("found",ctxOutput)
            #    continue

            #! where we save the 
            context = []
            #! walk on a trial folder of frames
            for root, dirs, files in os.walk(trialImageDir):
                for file in files:
                    if "frame" not in file:
                        continue 
                    
                    print(">", task_subject_trial, file+".txt" )
                    frameNumber = int(file.replace("frame_","").replace(".txt","").replace(".png",""))
                    #MP_comb = os.path.join(self.mpDir,task_subject_trial+".txt")                   

                    #! turn on for MPs
                    #MPI = MPInterface(MP_comb)
                    Context_comb = os.path.join(self.ContextDir,task_subject_trial+".txt")
                    Pred_Context_comb = os.path.join(self.context_output,task_subject_trial+".txt")
                    CtxI = ContextInterface(Pred_Context_comb,False)
                    CtxI_Pred = ContextInterface(Pred_Context_comb,False)

                    '''
                    If we replace "images" by "labels" then the image source should be the same as the label source,
                    which is the same as the output destination
                    '''
                    labelRoot = root.replace("images","annotations")
                    RgrasperRoot = root.replace("images","deeplab_grasper_R_v3")
                    LgrasperRoot = root.replace("images","deeplab_grasper_L_v3")
                    threadRoot = root.replace("images","deeplab_thread_v3")
                    ringRoot = root.replace("images","deeplab_rings_v3")
                    needleRoot = root.replace("images","deeplab_needle_v3")
                    outputRoot = root.replace("images","labeled_images")

                    imageSource = os.path.join(trialImageDir, file)
                    labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                    RgrasperMask = os.path.join(RgrasperRoot, self.imageToNPY(file))
                    LgrasperMask = os.path.join(LgrasperRoot, self.imageToNPY(file))
                    threadMask = os.path.join(threadRoot, self.imageToNPY(file))
                    ringMask = os.path.join(ringRoot,self.imageToNPY(file))
                    needleMask = os.path.join(needleRoot,self.imageToNPY(file))
                    
                    outputDest = os.path.join(outputRoot, file)

                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    #! if os.path.exists(outputDest): os.remove(outputDest)
                    if not os.path.exists(labelSource) and False:
                        print("label not found for ",imageSource)
                        #continue
                        pass
                    else:
                        #! we are drawing the image so late in the execution because we need to predict the context annotation first
                        if("Knot" in self.task):
                            #self.DrawSingleImageKT(imageSource,labelSource,outputDest)
                            #! LG_Thread_Info, RG_Thread_Info contains intersections between graspers and the nearest thread top or bottom (combined left and right)
                            #self, grasperSource, ThreadSource, target,CtxI,CtxI_Pred, DEBUG=False
                            LG_Thread_Info_Top,RG_Thread_Info_Top, LG_Thread_Info_Bottom,RG_Thread_Info_Bottom = self.DrawSingleImageContextKT(LgrasperMask,RgrasperMask,threadMask,outputDest, CtxI,CtxI_Pred)
                            
                        elif("Needle" in self.task):
                            #! LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info
                            #LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextNP(grasperMask,threadMask,needleMask,ringMask,outputDest, CtxI,CtxI_Pred)
                            LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,LG_inter_R,RG_inter_R,Ring_Needle_inter =  self.DrawSingleImageContextNP(grasperMask,threadMask,needleMask,ringMask,outputDest, CtxI,CtxI_Pred)
                        else:
                            #LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContextS(grasperMask,threadMask,needleMask,outputDest, CtxI,CtxI_Pred)
                            LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N = self.DrawSingleImageContextS(grasperMask,threadMask,needleMask,outputDest, CtxI,CtxI_Pred)
                    #! LG_Info[1] for intersection with needle
                    #! frameNumber, L_Gripper_Angle, R_Gripper_Angle
                    # we can then use the object Frame to determine the context
                    # Contact/Hold Context:
                    # "Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"
                    #           0                   1         2         3               4       5       6                   
                    if("Needle" in self.task):
                        # Needle State in Suturing:
                        # "Not Touching", "Touching", "In"
                        #               0       1       2
                        # Frame# {0,2,3}, {0,2,3}, {0,2,3}, {0,2,3}, {0}

                        # Needle States in Needle Passing
                        # "Out of", "Touching","In"
                        #       0           1     2
                        # Frame# {0,2,3,5},{0,2,3,5},{0,2,3,5},{0,2,3,5},{0,1,2}
                        L_Gripping,R_Gripping = self.isGripperClosed(frameNumber,trialKin_L,trialKin_R)                    
                        L_G_Touch = 0
                        L_G_Hold = 0
                        R_G_Touch = 0
                        R_G_Hold = 0
                        # Thread = 3 
                        if(LG_inter_T):
                            if(L_Gripping):
                                L_G_Hold = 3
                            else:
                                L_G_Touch = 3
                        if(RG_inter_T):
                            if(R_Gripping):
                                R_G_Hold = 3
                            else:
                                R_G_Touch = 3

                        # Ring = 5
                        if(LG_inter_R):
                            if(L_Gripping):
                                L_G_Touch = 5
                            else:
                                L_G_Hold = 5
                        if(RG_inter_R):
                            if(R_Gripping):
                                R_G_Touch = 0
                            else:
                                R_G_Hold = 3

                        # Needle = 2 
                        if(LG_inter_N):
                            if(L_Gripping):
                                L_G_Hold = 2
                                L_G_Touch = 0
                            else:
                                L_G_Hold = 0
                                L_G_Touch = 2
                        if(RG_inter_N):
                            if(R_Gripping):
                                R_G_Hold = 2
                                R_G_Touch = 0
                            else:
                                R_G_Hold = 0
                                R_G_Touch = 2

                        Extra_State = 0
                        if("Needle" in self.task):
                            if(Ring_Needle_inter):
                                Extra_State = 2
                            else:
                                Extra_State = 0
                        context.append(str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State))
                    elif("Suturing" in self.task):
                        L_Gripping,R_Gripping = self.isGripperClosed(frameNumber,trialKin_L,trialKin_R)                    
                        L_G_Touch = 0
                        L_G_Hold = 0
                        R_G_Touch = 0
                        R_G_Hold = 0
                        # Thread = 3 
                        if(LG_inter_T):
                            if(L_Gripping):
                                L_G_Hold = 3
                            else:
                                L_G_Touch = 3
                        if(RG_inter_T):
                            if(R_Gripping):
                                R_G_Hold = 3
                            else:
                                R_G_Touch = 3
                       
                        # Needle = 2 
                        if(LG_inter_N):
                            if(L_Gripping):
                                L_G_Hold = 2
                                L_G_Touch = 0
                            else:
                                L_G_Hold = 0
                                L_G_Touch = 2
                        if(RG_inter_N):
                            if(R_Gripping):
                                R_G_Hold = 2
                                R_G_Touch = 0
                            else:
                                R_G_Hold = 0
                                R_G_Touch = 2
                        Extra_State = 0
                        context.append(str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State))
                    else: #! Knot Tying                        
                        # Knot States in Knot Tying:
                        #"N/A", "Thread Wrapped", "Loose", "Tight"
                        #    0,                1,       2,       3
                        #L_Gripping,R_Gripping = self.isGripperClosed(frameNumber,trialKin_L,trialKin_R) 
                        L_Gripping,R_Gripping = True, True
                        #print("\t","LG_Thread_Info_Top",LG_Thread_Info_Top[1],"LG_Thread_Info_Bottom",LG_Thread_Info_Bottom[1],"RG_Thread_Info_Top",RG_Thread_Info_Top[1],"RG_Thread_Info_Bottom",RG_Thread_Info_Bottom[1])   
                        #print("\t",frameNumber,": L_Gripping",L_Gripping,"R_Gripping",R_Gripping)                   
                        L_G_Touch = 0
                        L_G_Hold = 0
                        R_G_Touch = 0
                        R_G_Hold = 0
                        if( (LG_Thread_Info_Top[1] or LG_Thread_Info_Bottom[1]) and L_Gripping ):
                            L_G_Touch = 0
                            L_G_Hold = 3
                        elif((LG_Thread_Info_Top[1] or LG_Thread_Info_Bottom[1]) and not L_Gripping ):
                            L_G_Touch = 3
                            L_G_Hold = 0 
                        if( (RG_Thread_Info_Top[1] or RG_Thread_Info_Bottom[1]) and R_Gripping ):
                            R_G_Touch = 0
                            R_G_Hold = 3
                        elif((RG_Thread_Info_Top[1] or RG_Thread_Info_Bottom[1]) and not R_Gripping):
                            R_G_Touch = 3
                            R_G_Hold = 0 
                        Extra_State = 0                                       
                        context.append(str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State))
                        #print("\t",str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State),"\n")  
                    count += 1
            if(len(context) > 2):
                pass
                print("saving",ctxOutput,context)
                count+=1
                self.save(ctxOutput,context)
        print(count,"images processed!")

    def GenerateContextLine(self,L_G_Touch,L_G_Hold,R_G_Touch,R_G_Hold,L_Gripping,R_Gripping,N_Intersection,frameNumber):
        '''
        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        if(LG_Info[1] and L_Gripping):
            L_G_Touch = 0
            L_G_Hold = 2
        elif(LG_Info[1] and not L_Gripping):
            L_G_Touch = 2
            L_G_Hold = 0 
        if(RG_Info[1] and R_Gripping):
            R_G_Touch = 0
            R_G_Hold = 2
        elif(RG_Info[1] and not R_Gripping):
            R_G_Touch = 2
            R_G_Hold = 0 
        Extra_State = 0
        if("Needle" in self.task):
            if(N_Intersection != 3):
                Extra_State = 2
            else:
                Extra_State = 0                    
        return str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State)
        '''

    def DrawLabelsContext(self):
            count = 0
            for root, dirs, files in os.walk(self.imagesDir):
                for file in files:
                    if "frame" not in file:
                        continue
                    #if "frame_1599" in file or "frame_1264" in file or "frame_0805" in file or "frame_1572" in file:
                    #    continue
                    #if "Suturing_S02_T05" not in os.path.basename(root):
                    #    continue
                    print("Proc:", os.path.basename(root),file+".txt" )
                    Bname = os.path.basename(root)

                    MP_comb = os.path.join(self.mpDir,Bname+".txt")
                    #print(MP_comb)
                    MPI = MPInterface(MP_comb) # turn on for MPs as well

                    Context_comb = os.path.join(self.ContextDir,Bname+".txt")
                    #print(MP_comb)
                    CtxI = ContextInterface(Context_comb)

                    '''
                    If we replace "images" by "labels" then the image source should be the same as the label source, 
                    which is the same as the output destination
                    '''
                    imageRoot = root
                    #labelRoot = self.getDirectory(root,"labels")
                    labelRoot = root.replace("images_pre","annotations_pre")
                    #outputRoot =  self.getDirectory(root,"output")
                    outputRoot = root.replace("images_pre","labeled_images")

                    imageSource = os.path.join(imageRoot, file)
                    labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                    outputDest = os.path.join(outputRoot, file)

                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #if os.path.exists(outputDest):
                    #    os.remove(outputDest)         

                    if not os.path.exists(labelSource):
                        #print("label not found for ",imageSource)
                        continue
                    else:
                        #self.DrawLabel(imageSource,labelSource,outputDest)
                        if("Knot" in self.task):
                            self.DrawSingleImageKT(imageSource,labelSource,outputDest)
                        else:
                            LG_Info, RG_Info, N_Intersection, Needle_Ring_Distances,LG_Thread_Info, RG_Thread_Info = self.DrawSingleImageContext(imageSource,labelSource,outputDest, MPI, CtxI)
                    
                    

                    # we can then use the object Frame to determine the context
                    # Contact/Hold Context:
                    # "Nothing", "Ball/Block/Sleeve", "Needle", "Thread", "Fabric/Tissue", "Ring", "Other"
                    #           0                   1         2         3               4       5       6


                    # Needle State in Suturing:
                    # "Not Touching", "Touching", "In"
                    #               0       1       2
                    # Frame# {0,2,3}, {0,2,3}, {0,2,3}, {0,2,3}, {0}

                    # Needle States in Needle Passing
                    # "Out of", "Touching","In"
                    #       0           1     2
                    # Frame# {0,2,3,5},{0,2,3,5},{0,2,3,5},{0,2,3,5},{0,1,2}

                    # Knot States in Knot Tying:
                    

                    count += 1
                    
            print(count,"images processed!")

    #! Needs: self.imagesDir, 
    def DrawLabels(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue
                #if "Suturing_S02_T05" not in os.path.basename(root):
                #    continue
                print("Proc:", os.path.basename(root),file+".txt" )
                Bname = os.path.basename(root)

                MP_comb = os.path.join(self.mpDir,Bname+".txt")
                print(MP_comb)
                MPI = MPInterface(MP_comb)


                '''
                If we replace "images" by "labels" then the image source should be the same as the label source, 
                which is the same as the output destination
                '''
                imageRoot = root
                #labelRoot = self.getDirectory(root,"labels")
                labelRoot = root.replace("images","annotations")
                #outputRoot =  self.getDirectory(root,"output")
                outputRoot = root.replace("images","vis_context_labels_gt")

                imageSource = os.path.join(imageRoot, file)
                labelSource = os.path.join(labelRoot, self.imageToJSON(file))
                outputDest = os.path.join(outputRoot, file)

                if(not os.path.isdir(outputRoot)):
                    path = pathlib.Path(outputRoot)
                    path.mkdir(parents=True, exist_ok=True)

                #if os.path.exists(outputDest):
                #    os.remove(outputDest)

                if not os.path.exists(labelSource):
                    #print("label not found for ",imageSource)
                    continue
                else:
                    #self.DrawLabel(imageSource,labelSource,outputDest)
                    if("Knot" in self.task):
                        self.DrawSingleImageKT(imageSource,labelSource,outputDest)
                    else:
                        self.DrawSingleImage(imageSource,labelSource,outputDest, MPI)

                count += 1
                
        print(count,"images processed!")

    def save(self, file, lines):
       
        with open(file, 'w+') as f:
            for item in lines:
                f.write("%s\n" % item)

main();



