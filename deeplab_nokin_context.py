import os, sys
import json
import pathlib
import math
from telnetlib import PRAGMA_HEARTBEAT
from tkinter import Pack
from turtle import position
from xmlrpc.client import Boolean
#from cv2 import KeyPoint, threshold
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
#from imantics import Polygons, Mask
import utils 
import rendering

import cv2

import matplotlib.pyplot as plt

import matplotlib.cm as cm

#from numba import jit, cuda


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
    task = "Suturing"
    I = Iterator(task)
    I.DrawDeepLab() #TODO: get ctx lines like consensus
    #I.GenerateContext()


    #I.DrawLabelsContext()
    quit();    
           
class Iterator:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")
        self.cogitoDir = os.path.join(self.CWD, task,"cogito_annotations")
        self.cogitoOutputDir = os.path.join(self.CWD, task,"cogito_labeled_images")
        self.context_output = os.path.join(self.CWD,task,"vis_context_labels_v4")        
        self.deeplabOutputDir= os.path.join(self.CWD,task,"deeplab_labeled_images")     
        self.ctxConsensusDir = os.path.join(self.CWD,task,"ctx_consensus")   
        self.ctxSurgeonDir = os.path.join(self.CWD,task,"ctx_surgeon")   
        self.ctxPredDir =  os.path.join(self.CWD,task,"ctx_output")   
        self.tissueDir =  os.path.join(self.CWD,task,"tissue_keypoints")
        self.grasperJawDir = os.path.join(self.CWD,task,"grasper_jaw_keypoints")

        self.OS = "windows"   
        
    ''' Iterator: DrawSingleImageKT(imageSource, labelSource, target, DEBUG=False)

    :param imageSource: FILEPATH - Image path
    :param labelSource: FILEPATH - used to get image segmentation annotation JSON files
    :param target:      FILEPATH - 
    :param DEBUG:       boolean - enable console printout

    :return: nothing'''     
    def DrawSingleImageKT(self, imageSource, labelSource, target, DEBUG=False):        
        J = utils.JSONInterface(labelSource) 
        polyNames , polygons = J.getPolygons(); #! graspers only in KT, 
        kpNames, KeyPoint = J.getKeyPoints(); #! None in KT,
        polyLineNames, polyLines = J.getPolyLines();
        if self.OS == "windows":
            font = ImageFont.truetype("arial.ttf", 12)
        else: 
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 14, encoding="unic")
        img = Image.open(imageSource)
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
            rendering.DrawThread(polyLines_TL, polyNames_TL, kp, draw, font)
        if(len(polyLines_TR)>0):                
            kp = [polyLines_TR[0][0], polyLines_TR[0][1]]             
            rendering.DrawThread(polyLines_TR, polyNames_TR, kp, draw, font)
        if(len(polyLines_BL)>0):                
            kp = [polyLines_BL[0][0], polyLines_BL[0][1]]             
            rendering.DrawThread(polyLines_BL, polyNames_BL, kp, draw, font)
        if(len(polyLines_BR)>0):                
            kp = [polyLines_BR[0][0], polyLines_BR[0][1]]             
            rendering.DrawThread(polyLines_BR, polyNames_BR, kp, draw, font)
        img.save(target)
        return 
    
    def DrawSingleImage(self, imageSource, labelSource, target, CtxI, DEBUG=False):
        J = utils.JSONInterface(labelSource)
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
        #def DrawPolygons(self, polygons,polyNames,draw,font):
        rendering.DrawPolygons(polygons,polyNames,draw,font)
        rendering.DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font)
        rendering.DrawTextTopCorner(CtxI.getContext(IDX), draw, font)
        #distances = self.CalcDistances(polygons,polyNames,polyLines,polyLineNames)
        if("Needle End" not in kpNames):
            #print("No needle for", imageSource)
            pass
        else: 
            needleEnd = KeyPoint[0]
            for i in range(len(KeyPoint)):
                if("Needle End" in kpNames[i]):
                    needleEnd = KeyPoint[i]
            if(len(polyLines) !=0):     
                rendering.DrawThread(polyLines, polyLineNames, needleEnd, draw, font)
        img.save(target) # to save
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
            LG_Points.append(  (LGX[i],LGY[i]))
        for i in range(len(RGX)):
            RG_Points.append(  (RGX[i], RGY[i]))
        for i in range(len(ThreadX)):
            Thread_Points.append(  (ThreadX[i], ThreadY[i]))
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
                RingPoints[j].append((RingsX[j][i],RingsY[j][i])  )

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
        Rgrasper = utils.NPYInterface2()
        try:
            (x_p_R,y_p_R), (i_R,j_R), interR = Rgrasper.getIntersectionFast(RgrasperMask, ThreadSource) # (center), (intersection), boolean
        #print((x_p,y_p), inter)
        #Lgrasper = NPYInterface2()
            (x_p_L,y_p_L), (i_L,j_L), interL = Rgrasper.getIntersectionFast(LgrasperMask, ThreadSource) # (center), (intersection), boolean
        except Exception:
            interR = False
            interL = False
        LG_inter_t = False
        RG_inter_t = False
        LG_inter_t = interL
        RG_inter_t = interR
        
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

    def DrawSingleImageContextNP(self, LgrasperMask,RgrasperMask,ThreadSource,needleMask,ringMask,outputDest, CtxI,CtxI_Pred):
        #J = JSONInterface(labelSource)
        Rgrasper = utils.NPYInterface2()
        try:
            #(x_p_T,y_p_T), (i_T,j_T), inter_T = prednpyG_T.getIntersection_cached(grasper_gt,grasper, threadMask) # (center), (intersection), boolean
            [grasper_gt,Lgrasper_n] = np.load(RgrasperMask,allow_pickle=True)
            [grasper_gt,Rgrasper_n] = np.load(LgrasperMask,allow_pickle=True)
            [grasper_gt,thread_n] = np.load(ThreadSource,allow_pickle=True)
            [grasper_gt,needle_n] = np.load(needleMask,allow_pickle=True)
            [grasper_gt,ring_n] = np.load(ringMask,allow_pickle=True)
            Lgrasper_n[Lgrasper_n>0.95]=1 #! instead of 0.97
            Lgrasper_n[Lgrasper_n<0.95]=0 #! instead of 0.97
            Rgrasper_n[Rgrasper_n>0.95]=1 
            Rgrasper_n[Rgrasper_n<0.95]=0 
            thread_n[thread_n>0.95]=1 
            thread_n[thread_n<0.95]=0 
            needle_n[needle_n>0.95]=1 
            needle_n[needle_n<0.95]=0 
            ring_n[ring_n>0.95]=1 
            ring_n[ring_n<0.95]=0 
            L_grasper = np.squeeze(Lgrasper_n)
            R_grasper = np.squeeze(Rgrasper_n)
            thread = np.squeeze(thread_n)
            needle = np.squeeze(needle_n)
            ring = np.squeeze(ring_n)

            #(x_p_R,y_p_R), (i_R,j_R), inter_T_R = Rgrasper.getIntersectionFastest(RgrasperMask, ThreadSource) # (center), (intersection), boolean
            #(x_p_L,y_p_L), (i_L,j_L), inter_T_L = Rgrasper.getIntersectionFastest(LgrasperMask, ThreadSource) # (center), (intersection), boolean
            inter_T_R, (i,j) = Rgrasper.isIntersecting(R_grasper,thread)
            inter_T_L, (i,j) = Rgrasper.isIntersecting(L_grasper,thread)


            #(x_p_N,y_p_N), (i_N,j_N), inter_N = prednpyG_N.getIntersection_cached(grasper_gt,grasper, needleMask) 
            #(x_p_R,y_p_R), (i_R,j_R), inter_N_R = Rgrasper.getIntersectionFastest(RgrasperMask, needleMask) # (center), (intersection), boolean
            #(x_p_L,y_p_L), (i_L,j_L), inter_N_L = Rgrasper.getIntersectionFastest(LgrasperMask, needleMask) # (center), (intersection), boolean
            inter_N_R, (i,j) = Rgrasper.isIntersecting(R_grasper,needle)
            inter_N_L, (i,j) = Rgrasper.isIntersecting(L_grasper,needle)

            #(x_p_R,y_p_R), (i_R,j_R), inter_R = prednpyG_R.getIntersection_cached(grasper_gt,grasper, ringMask) 
            #(x_p_R,y_p_R), (i_R,j_R), inter_R_R = Rgrasper.getIntersectionFastest(RgrasperMask, ringMask) # (center), (intersection), boolean
            #(x_p_L,y_p_L), (i_L,j_L), inter_R_L = Rgrasper.getIntersectionFastest(LgrasperMask, ringMask) # (center), (intersection), boolean
            inter_R_R, (i,j) = Rgrasper.isIntersecting(R_grasper,ring)
            inter_R_L, (i,j) = Rgrasper.isIntersecting(L_grasper,ring)

            #(x_p_R_N,y_p_R_N), (i_R_N,j_R_N), inter_R_N = prednpyR_N.getIntersection(needleMask, ringMask) 
            #(x_p_L,y_p_L), (i_L,j_L), inter_R_N = Rgrasper.getIntersectionFastest(needleMask, ringMask) # (center), (intersection), boolean
            inter_R_N, (i,j) = Rgrasper.isIntersecting(needle,ring)
        #Lgrasper = NPYInterface2()
        #print((x_p,y_p), inter)
        except Exception:
            inter_T_R,inter_T_L,inter_N_R,inter_N_L,inter_R_R,inter_R_L,inter_R_N = False,False,False,False,False,False,False

        LG_inter_T = inter_T_L
        RG_inter_T = inter_T_R        

        LG_inter_N = inter_N_L
        RG_inter_N = inter_N_R
                
        LG_inter_R = inter_R_L
        RG_inter_R = inter_R_R
                
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
        
    def makeLenStr(self,d1,d2,d3,d4):
        if(len(d1)) != str(len(d2)) != (len(d3)) != (len(d4)) != 480:
            return str(len(d1))+"="+str(len(d2))+"="+str(len(d3))+"="+str(len(d4))
        else: "no match" 
    def makeLenStr0(self,d1,d2,d3,d4):
        if(len(d1[0])) != str(len(d2[0])) != (len(d3[0])) != (len(d4[0])):
            return str(len(d1[0]))+"="+ str(len(d2[0]))+"="+ str(len(d3[0]))+"="+ str(len(d4[0]))
    
    def distancesWithinThreshold(self,kp, pointArr, thresh):
        minDist = 10000
        inter = False
        for point in pointArr:
            dist = math.hypot(point[0]-kp[0],point[1]-kp[1])
            if(dist < minDist):
                minDist = dist
            if(dist<thresh):
               inter = True
        return inter, minDist

    def HorizontalDistancesWithinThreshold(self,kp, pointArr, thresh):
        minDist = 10000
        inter = False
        index = 0
        while(index < len(pointArr)):
            dist1 = math.hypot(pointArr[index][0]-kp[0],pointArr[index][1]-kp[1])
            index = index+1
            dist2 = math.hypot(pointArr[index][0]-kp[0],pointArr[index][1]-kp[1])
            index = index+1
            avg = np.average([dist1,dist2])
            if(avg < minDist):
                minDist = avg
            if(avg<thresh):
               inter = True
        return inter, minDist
    def distTwoPoints(seld,p1,p2):
        return math.hypot(p1[0]-p2[0],p1[1]-p2[1])

    def getTissueBisector(self,Points):
        c1 = Points[0]
        c2 = Points[1]

        c3 = Points[-2]
        c4 = Points[-1]

        return [(c1[0]+c2[0])/2,(c1[1]+c2[1])/2],[(c3[0]+c4[0])/2,(c3[1]+c4[1])/2] 


    #@jit(target_backend='cuda') 
    def SuturingInter(self,L_grasper,R_grasper,Thread,Needle,TissuePoints):
        #norm_rows = cuda.grid(1)
        #norm_cols = cuda.grid(1)
        norm_rows = 480
        norm_cols = 640
        LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping = False, False, False, False, False,False, False
        # Possible expansion: now you have annotations for the grasper jaws
        min_Tissue_Dist = 10000
        min_needle_center_dist = 10000
        for r in range(norm_rows):
            for c in range(norm_cols):
                LG = L_grasper[r][c]
                RG = R_grasper[r][c]
                T = Thread[r][c]
                N = Needle[r][c]
                if(LG==1):
                    if(N==1):
                        LG_inter_N = True
                    elif(T==1):
                        LG_inter_T = True
                if(RG==1):
                    if(N==1):
                        RG_inter_N = True
                    elif(T==1):
                        RG_inter_T = True
                if(N==1):
                    kp = [c,r]
                    #N_inter,min_Tissue_Dist_point = self.HorizontalDistancesWithinThreshold(kp,TissuePoints,10)
                    N_inter,min_Tissue_Dist_point = self.distancesWithinThreshold(kp,TissuePoints,10)
                    
                    
                    p3=np.array([c,r])
                    P1,P2 = self.getTissueBisector(TissuePoints)
                    p1=np.array(P1)
                    p2=np.array(P2)

                    if N_inter:
                        N_inter_TS = True
                    if min_Tissue_Dist_point < min_Tissue_Dist:
                        min_Tissue_Dist = min_Tissue_Dist_point
                    
                    min_needle_center_dist_point = np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
                    if min_needle_center_dist_point < min_needle_center_dist:
                        min_needle_center_dist = min_needle_center_dist_point
                    #distance form a point to a line
                    

        return LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,min_Tissue_Dist,min_needle_center_dist

    def GetSuturingIntersections(self, L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred):
        
        lines = []
        norm_rows = 480
        norm_cols = 640
        #print("Rows:",self.makeLenStr(L_grasper,R_grasper,Thread,Needle),"\nCols:",self.makeLenStr0(L_grasper,R_grasper,Thread,Needle))    
            
        #if(len(L_grasper) != norm_rows or len(R_grasper) != norm_rows or len(Thread) != norm_rows or len(Needle) != norm_rows ):
        #    print("Row mismatch",self.makeLenStr(L_grasper,R_grasper,Thread,Needle))
        #if(len(L_grasper[0]) != norm_cols or len(R_grasper[0]) != norm_cols or len(Thread[0]) != norm_cols or len(Needle[0]) != norm_cols ):
        #    print("Col mismatch",self.makeLenStr0(L_grasper,R_grasper,Thread,Needle))

        LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,min_Tissue_Dist,needle_center_dist = self.SuturingInter(L_grasper,R_grasper,Thread,Needle,TissuePoints)
        
        try:
            L_Dist = self.distTwoPoints(GrasperJawPoints[2],GrasperJawPoints[3])
            R_Dist = self.distTwoPoints(GrasperJawPoints[0],GrasperJawPoints[1])
        except Exception as e:
            L_Dist = 30
            R_Dist = 30 
        PARAM_JAW_DIST = 20
        if(L_Dist < PARAM_JAW_DIST):
            L_Gripping = True
        if(R_Dist < PARAM_JAW_DIST):
            R_Gripping = True
        
        return LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist

    def GenerateContextLineS(self,LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,min_Tissue_Dist,needle_center_dist,frameNumber):

        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        Extra_State = 0

        if(R_Gripping):            
            if(RG_inter_N):
                R_G_Hold = 2
            elif(RG_inter_T):
                R_G_Hold = 3
        else: #Right not gripping
            if(RG_inter_N):
                R_G_Touch = 2
            elif(RG_inter_T):
                R_G_Touch = 3

        if(L_Gripping):            
            if(LG_inter_N):
                L_G_Hold = 2
            elif(LG_inter_T):
                L_G_Hold = 3
        else:#Left not gripping
            if(LG_inter_N):
                L_G_Touch = 2
            elif(LG_inter_T):
                L_G_Touch = 3
        if(N_inter_TS):
            if(R_G_Hold == 2 and L_G_Hold == 2):
                Extra_State = 0
            if(R_G_Hold == 2):
                Extra_State = 1 #if needle end is near the tissue
            if(L_G_Hold == 2):
                Extra_State = 1 #if needle end is near the tissue  
        if(R_G_Hold == 2 and L_G_Hold == 2 and min_Tissue_Dist < 5 and needle_center_dist > 20):
            Extra_State = 2

        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State)

        
    #L_grasper,R_grasper,thread,needle,TissuePoints,GrasperJawPoints,outputFName,CtxI,CtxI_Pred
    def DrawSingleImageContextS(self, L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred,ctxPredLine,frameNumber,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist):
        fig, ax = plt.subplots( nrows=1, ncols=1 )        
        image = cv2.imread(imageFName)
        plt.imshow(image, cmap='gray') 

        masked_L_grasper = np.ma.masked_where(L_grasper < 0.9, L_grasper)
        masked_R_grasper = np.ma.masked_where(R_grasper < 0.9, R_grasper)
        masked_Thread = np.ma.masked_where(Thread < 0.9, Thread)
        masked_Needle = np.ma.masked_where(Needle < 0.9, Needle)
        ax.imshow(masked_L_grasper, cmap='Spectral', alpha=0.5,interpolation=None) # I would add 
        ax.imshow(masked_R_grasper, cmap='Spectral', alpha=0.5,interpolation=None)        
        ax.imshow(masked_Thread, cmap='spring', alpha=0.5,interpolation=None)        
        ax.imshow(masked_Needle, cmap='cool', alpha=0.5,interpolation=None)        
        i=1
        try:
            for point in TissuePoints:            
                drawObject = plt.Circle(point,radius=2,color='red', fill=True)
                ax.text(point[0],point[1],str(i),fontsize=8,color='red')            
                ax.add_patch(drawObject)
                i=i+1

            P1,P2 = self.getTissueBisector(TissuePoints)  #[(c1[0]+c2[0])/2,(c1[1]+c2[1])/2],[(c3[0]+c4[0])/2,(c3[1]+c4[1])/2] 
            drawObject = plt.Circle(P1,radius=2,color='green', fill=True)
            ax.add_patch(drawObject)
            drawObject = plt.Circle(P2,radius=2,color='green', fill=True)
            ax.add_patch(drawObject)          

            
        except Exception as e:
            print(e)
            return

        try:
                L_Grasper_Points = [GrasperJawPoints[2],GrasperJawPoints[3]]
                R_Grasper_Points = [GrasperJawPoints[0],GrasperJawPoints[1]]

                
                for point in L_Grasper_Points:
                    drawObject = plt.Circle(point,radius=2,color='red', fill=True)
                    ax.add_patch(drawObject)
                for point in R_Grasper_Points:
                    drawObject = plt.Circle(point,radius=2,color='green', fill=True)
                    ax.add_patch(drawObject)
        except Exception as e:
            pass
                
        #inter, (i,j)  = utils.isIntersecting(L_grasper,Needle)
        #CtxI.getContext()        "PRED:"
        try:
            rendering.DrawTextArrPlt(["     GT:"+CtxI.getContext(frameNumber),"PRED:"+ctxPredLine,"Jaw Dist L:"+str(L_Dist),"Jaw Dist R:"+str(R_Dist),"TS Dist:"+str(min_Tissue_Dist),"N Dist:"+str(needle_center_dist)],ax)
        except Exception as e:
            print(e)
            return
        '''
        inter_msg = ""
        if(inter):
            inter_msg = "Intersection: True"
        else:
            inter_msg = "Intersection: False"
        ax.text(50,50,inter_msg,fontsize=30,color='red')
        '''
        fig.savefig(outputFName) 
        plt.close(fig)   

    
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
    def DrawSingleImageContext(self, imageSource, labelSource,predicted_labelSource, target, CtxI,CtxI_Pred, DEBUG=False):       
        J = utils.JSONInterface(labelSource)
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


    def DrawLabelsWithContext(self):        
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue

                imageRoot = root                
                imageSource = os.path.join(imageRoot, file)
                #cogitoRoot = root.replace("images","cogito_annotations")                
                cogitoSource = os.path.join(self.cogitoDir, utils.imageToJSON(file))
                #outputRoot =  self.getDirectory(root,"output")
                outputDest = os.path.join(self.outputDir, file)
                ctxSource = os.path.join(self.cogitoDir, utils.imageToTXT(file))


    def GenerateContext(self):
        count = 0
        Tissues = {} 
        GrasperJaws = {}
        for root, dirs, files in os.walk(self.tissueDir):
            for file in files:
                #if "frame" not in file:
                tissueFname = os.path.join(root,file)
                Tissue = utils.ViaJSONInterface(tissueFname).getDataDict()
                Tissues[file.replace(".json","")] = Tissue

        for root, dirs, files in os.walk(self.grasperJawDir):
            for file in files:
                #if "frame" not in file:
                jawsFname = os.path.join(root,file)
                GrasperJaw = utils.ViaJSONInterface(jawsFname).getDataDict()
                GrasperJaws[file.replace(".json","")] = GrasperJaw
        #for k, v in Tissues.items():
            #print(k, v)
        
        Dirs = []
        oldRoot = ""
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            oldRoot = root
            break
        print("Trials:",Dirs)
        for Trial in Dirs:
            TrialRoot = os.path.join(self.imagesDir,Trial)
            contextLines = []
            ctxFName = os.path.join(self.ctxConsensusDir, Trial+".txt")  
            ctxPredFName = os.path.join(self.context_output,Trial+".txt")  
            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if "frame" not in file:
                        continue
                    
                    
                    imageRoot = root 
                    frameNumber = int(file.replace(".png","").split("_")[1])        
                    #if( frameNumber< 5628):continue            
                    trialFname = os.path.basename(root)
                    imageFName = os.path.join(imageRoot, file)
                    #cogitoRoot = root.replace("images","cogito_annotations")                
                    cogitoFName = os.path.join(self.cogitoDir,trialFname, utils.imageToJSON(file))
                    outputRoot =  os.path.join(self.deeplabOutputDir,trialFname)
                    outputFName = os.path.join(self.deeplabOutputDir,trialFname, file)
                    #print("outputFName",outputFName)
                                
                    
                    TissuePoints = Tissues[trialFname][file]
                    GrasperJawPoints = GrasperJaws[trialFname][file]                    
                    
                    #print("Drawing Segmentation labels and Keypoints:", os.path.basename(root),file)

                    RgrasperRoot = root.replace("images","deeplab_grasper_R_v4")
                    LgrasperRoot = root.replace("images","deeplab_grasper_L_v4")
                    threadRoot = root.replace("images","deeplab_thread_v4")
                    ringRoot = root.replace("images","deeplab_rings_v4")
                    needleRoot = root.replace("images","deeplab_needle_v4")

                    RgrasperMask = os.path.join(RgrasperRoot, utils.imageToNPY(file))
                    LgrasperMask = os.path.join(LgrasperRoot, utils.imageToNPY(file))
                    threadMask = os.path.join(threadRoot, utils.imageToNPY(file))
                    ringMask = os.path.join(ringRoot, utils.imageToNPY(file))
                    needleMask = os.path.join(needleRoot, utils.imageToNPY(file))

                    L_grasper = utils.NPYInterface3.loadArr(LgrasperMask)
                    R_grasper = utils.NPYInterface3.loadArr(RgrasperMask)
                    Thread = utils.NPYInterface3.loadArr(threadMask)
                    Needle = utils.NPYInterface3.loadArr(needleMask)
                    #ring = utils.NPYInterface3.loadArr(ringMask)
                        
                    #MPI = MPInterface(MP_comb) # turn on for MPs as well
                    CtxI = utils.ContextInterface2(ctxFName)
                    CtxI_Pred = utils.ContextInterface(ctxFName,True)
                    
                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)                
                    if("Knot" in self.task):
                        LG_Thread_Info_Top,RG_Thread_Info_Top, LG_Thread_Info_Bottom,RG_Thread_Info_Bottom = self.DrawSingleImageContextKT(
                            LgrasperMask,RgrasperMask,threadMask,outputFName, CtxI,CtxI_Pred)
                    if("Needle" in self.task):
                        LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,LG_inter_R,RG_inter_R,Ring_Needle_inter =  self.DrawSingleImageContextNP(LgrasperMask,RgrasperMask,threadMask,needleMask,ringMask,outputFName, CtxI,CtxI_Pred)
                    if("Suturing" in self.task):
                        LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist = self.GetSuturingIntersections(L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred)
                        ctxPredLine = self.GenerateContextLineS(LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,min_Tissue_Dist,needle_center_dist,frameNumber)
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        self.DrawSingleImageContextS(L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred,ctxPredLine,frameNumber,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist)
                    count += 1   
            if(len(contextLines) > 2):                
                print("saving",ctxPredFName)
                count+=1
                utils.save(ctxPredFName,contextLines)             
        print(count,"images processed!")

    def DrawDeepLab(self):
        count = 0
        Tissues = {} 
        GrasperJaws = {}
        for root, dirs, files in os.walk(self.tissueDir):
            for file in files:
                #if "frame" not in file:
                tissueFname = os.path.join(root,file)
                Tissue = utils.ViaJSONInterface(tissueFname).getDataDict()
                Tissues[file.replace(".json","")] = Tissue

        #for k, v in Tissues.items():
            #print(k, v)
        
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        for Trial in Dirs:
            TrialRoot = os.path.join(self.imagesDir,Trial)
            contextLines = []
            ctxFName = os.path.join(self.ctxConsensusDir, Trial+".txt")  
            ctxPredFName = os.path.join(self.context_output,Trial+".txt")  
            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if "frame" not in file:
                        continue
                    
                    
                    imageRoot = root 
                    frameNumber = int(file.replace(".png","").split("_")[1])        
                    #if( frameNumber< 5628):continue            
                    trialFname = os.path.basename(root)
                    imageFName = os.path.join(imageRoot, file)
                    #cogitoRoot = root.replace("images","cogito_annotations")                
                    cogitoFName = os.path.join(self.cogitoDir,trialFname, utils.imageToJSON(file))
                    outputRoot =  os.path.join(self.deeplabOutputDir,trialFname)
                    outputFName = os.path.join(outputRoot, file)
                    #print("outputFName",outputFName)
                                
                    
                    TissuePoints = Tissues[trialFname][file]
                    
                    
                    #print("Drawing Segmentation labels and Keypoints:", os.path.basename(root),file)

                    RgrasperRoot = root.replace("images","deeplab_grasper_R_v4")
                    LgrasperRoot = root.replace("images","deeplab_grasper_L_v4")
                    threadRoot = root.replace("images","deeplab_thread_v4")
                    ringRoot = root.replace("images","deeplab_rings_v4")
                    needleRoot = root.replace("images","deeplab_needle_v4")

                    RgrasperMask = os.path.join(RgrasperRoot, utils.imageToNPY(file))
                    LgrasperMask = os.path.join(LgrasperRoot, utils.imageToNPY(file))
                    threadMask = os.path.join(threadRoot, utils.imageToNPY(file))
                    ringMask = os.path.join(ringRoot, utils.imageToNPY(file))
                    needleMask = os.path.join(needleRoot, utils.imageToNPY(file))

                    L_grasper = utils.NPYInterface3.loadArr(LgrasperMask)
                    R_grasper = utils.NPYInterface3.loadArr(RgrasperMask)
                    Thread = utils.NPYInterface3.loadArr(threadMask)
                    Needle = utils.NPYInterface3.loadArr(needleMask)
                    #ring = utils.NPYInterface3.loadArr(ringMask)
                        
                    #MPI = MPInterface(MP_comb) # turn on for MPs as well
                    CtxI = utils.ContextInterface2(ctxFName)
                    CtxI_Pred = utils.ContextInterface(ctxFName,True)
                    
                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)                
                    if("Knot" in self.task):
                        #LG_Thread_Info_Top,RG_Thread_Info_Top, LG_Thread_Info_Bottom,RG_Thread_Info_Bottom = self.DrawSingleImageContextKT(
                        #LgrasperMask,RgrasperMask,threadMask,outputFName, CtxI,CtxI_Pred)
                        pass
                    if("Needle" in self.task):
                        #LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,LG_inter_R,RG_inter_R,Ring_Needle_inter =  self.DrawSingleImageContextNP(LgrasperMask,RgrasperMask,threadMask,needleMask,ringMask,outputFName, CtxI,CtxI_Pred)
                        pass
                    if("Suturing" in self.task):
                        ctxPredLine = ""
                        L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist = 1,1,1,1
                        self.DrawSingleImageContextS(L_grasper,R_grasper,Thread,Needle,TissuePoints,[],imageFName,outputFName,CtxI,CtxI_Pred,ctxPredLine,frameNumber,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist)
                    count += 1   
            if(len(contextLines) > 2):                
                print("saving",ctxPredFName)
                count+=1
                utils.save(ctxPredFName,contextLines)             
        print(count,"images processed!")

    def DrawVisLabels(self):
        count = 0
        for root, dirs, files in os.walk(self.imagesDir):
            for file in files:
                if "frame" not in file:
                    continue

                imageRoot = root         
                trialFname = os.path.basename(root)
                imageFName = os.path.join(imageRoot, file)
                #cogitoRoot = root.replace("images","cogito_annotations")                
                cogitoFName = os.path.join(self.cogitoDir,trialFname, utils.imageToJSON(file))
                #outputRoot =  self.getDirectory(root,"output")
                outputFName = os.path.join(self.cogitoOutputDir,trialFname, file)
                print("outputFName",outputFName)
                ctxFName = os.path.join(self.cogitoDir, trialFname,utils.imageToTXT(file))
                
                print("Drawing Segmentation labels and Keypoints:", os.path.basename(root),file)
                    
                #MPI = MPInterface(MP_comb) # turn on for MPs as well
                CtxI = utils.ContextInterface(ctxFName,True)

                if(not os.path.isdir(self.cogitoOutputDir)):
                    path = pathlib.Path(self.cogitoOutputDir)
                    path.mkdir(parents=True, exist_ok=True)
                #if os.path.exists(outputDest):
                #    os.remove(outputDest)
                if("Knot" in self.task):
                    self.DrawSingleImageKT(imageFName,cogitoFName,outputFName)
                else:                    
                    self.DrawSingleImage(imageFName,cogitoFName,outputFName,CtxI)
                    #self.DrawSingleImageContext(imageSource,labelSource,outputDest, MPI, CtxI)
                count += 1                
        print(count,"images processed!")

main();



