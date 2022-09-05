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
#from imantics import Polygons, Mask
import utils 

# list of colors for the annotations
colors =["#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]

# opacity of the annotation masks. Values range from (0 to 255) Type tuple
opacity = (180,)
# radius of keypoint
radius = 3

def getRBGA( hexColor):
    c = ImageColor.getcolor(hexColor, "RGB")        
    c = c + opacity
    return c

def RenderThread_Arr(thread_X, thread_Y, draw, font):        
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


def distNeedle(ax, ay,needleEnd ):
    return math.sqrt( (ax - needleEnd[0] )**2+( ay - needleEnd[1] )**2 )

#! PutTheadInOrder(polyLines arr,needleEnd as keypoint)
def PutTheadInOrder(polyLines_M,kp):
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
            thread_idx_0_distances.append(distNeedle(fx,fy,kp))
            thread_idx_1_distances.append(distNeedle(lx,ly,kp))       

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


def DrawThread(polyLines, polyLineNames, needleEnd, draw, font):  
    thread_X = []
    thread_Y = []
    if(len(polyLines)<2):
        for i in range(len(polyLines)):
            l = len(polyLines)
            for j in range(0,len(polyLines[i]),2):
                thread_X.append(polyLines[i][j])
                thread_Y.append(polyLines[i][j+1])
    else:
        thread_X, thread_Y = PutTheadInOrder(polyLines,needleEnd)      

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
        c = getRBGA(colors[-i])
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

    ThreadX, ThreadY = RenderThread_Arr(thread_X, thread_Y, draw, font)
    return ThreadX, ThreadY
        
def DrawKeyPoints(KeyPoint, kpNames, polygons,draw,font):
    for i in range(len(KeyPoint)): # draws each KeyPoint
        x = KeyPoint[i][0]
        y = KeyPoint[i][1]            
        leftUpPoint = (x-radius, y-radius)
        rightDownPoint = (x+radius, y+radius)
        twoPointList = [leftUpPoint, rightDownPoint]
        c = getRBGA(colors[i+(len(polygons))])
        draw.ellipse(twoPointList, fill=c)
        draw.text((x-radius*2, y-radius),kpNames[i]+str(i),(255,255,255),font=font)

def DrawTextTopCorner(MPI_str,draw,font):
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

def DrawTextArr(strArr,draw,font):
    if(len(strArr) == 0 or strArr[0] is None): 
        return
    offset = 1
    for s in strArr:
        x = 10
        y = 15 * offset; 
        draw.text( (x,y),s,(255,255,255),font=font);
        offset+=1

def DrawTextArrPlt(strArr,ax,):
    if(len(strArr) == 0 or strArr[0] is None): 
        return
    offset = 1
    for s in strArr:
        x = 10
        y = 22 * offset; 
        ax.text(x,y,s,fontsize=12,color='red')
        offset+=1

def DrawPolygons(polygons,polyNames,draw,font):
    for i in range(len(polygons)):
        #if("Ring" in polyNames[i]):
        c = getRBGA(colors[i])
            #print("Poly1:",polygons[i])
        draw.polygon(polygons[i], fill=c) #,outline='#EA5536')     
            ########## CENTER POINT
        x_c, y_c = Centroid(polygons[i])          
        leftUpPoint = (x_c-radius, y_c-radius)
        rightDownPoint = (x_c+radius, y_c+radius)
        twoPointList = [leftUpPoint, rightDownPoint]
        c = getRBGA(colors[i+(len(polygons))])
        draw.ellipse(twoPointList,outline=1, fill=c)            
            # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((x_c-radius*2, y_c-radius),polyNames[i]+str(i),(255,255,255),font=font)

def Centroid( points):
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