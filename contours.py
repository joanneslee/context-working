import numpy as np
import cv2 as cv
import os
import utils
import pathlib


#Find Contours:
#C:\Users\ianre\Desktop\coda\context-prediction-main\Needle_Passing\deeplab_grasper_L_v3\Needle_Passing_S04_T01frame_0046_pred.png
colors =["#E36D6D","#5E81B5","#D47BC9","#7CEB8E","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]

def main():
    #path = 'C:\\Users\\ianre\\Desktop\\coda\\context-prediction-main\\Needle_Passing\\deeplab_grasper_L_v3\\Needle_Passing_S04_T01'
    
    task = "Knot_Tying"
    I = Iterator(task)
    label_classes = ["deeplab_grasper_L_v3","deeplab_grasper_R_v3","deeplab_thread_v3"] #,"deeplab_needle_v3"
    label_classNames = ["dl_grasper_L","dl_grasper_R","dl_thread"] # ,"dl_needle"
    j=0
    for label_class in label_classes:
        I.findAllContours(label_class,label_classNames[j])
        j+=1
        #pass
    
    #I.findRingContours("deeplab_rings_v3","ring_X")



class Iterator:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")    
        self.deeplabOutputDir= os.path.join(self.CWD,task,"deeplab_labeled_images")
        self.tissueDir =  os.path.join(self.CWD,task,"tissue_keypoints")
        self.grasperJawDir = os.path.join(self.CWD,task,"grasper_jaw_keypoints")
        self.OS = "windows" 


    def idRing(self, cx, cy):
        points = [ [185,207],[290,213],[394,206],[497,236]]
        closestIndex = -1
        closestDist = 10000
        for i in range(len(points)):
            p = points[i]
            d =utils.distTwoPoints([cx,cy],p)
            if d < closestDist:
                closestDist = d
                closestIndex = i
        return str(closestIndex+4),closestIndex

    def findRingContours(self,LabelClass,LabelClassName):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 

                    #RegionAttributes = ["Ring_4","Ring_5","Ring_6","Ring_7"]
                    RegionAttributes = []
                    Regions = []
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        
                        M = cv.moments(cnt)
                        print( M )
                        try:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                        except Exception as e:
                            print(e,"weird moment error")
                            continue
                        ringID,closestIndex = self.idRing(cx,cy)                        
                        #Rcontours[closestIndex].append(cnt)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                        X = []
                        Y = []
                        epsilon = 0.01*cv.arcLength(cnt,True)
                        approx = cv.approxPolyDP(cnt,epsilon,True)
                        for points in approx:
                            x =int(points[0][0])
                            y = int(points[0][1])
                            X.append(x)
                            Y.append(y)
                        RegionAttributes.append("Ring_"+ringID)
                        Regions.append([X,Y])

                    #ringIDs = ["Ring_4","Ring_5","Ring_6","Ring_7"]

                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))        
                    #cnt = contours[largestIndex]
                    
                    ##### DRAW
                    #cv.drawContours(im,[approx],0,rbg,1)
                    #cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                    
                    #rbg = tuple(int(colors[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    #cv.drawContours(im,approx,0,rbg,thickness)
                    
                    

                    #Regions = [[X,Y],[X,Y]]
                    VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
                    #VIA.addRings(file, LabelClassName, PolyPointsY)
                    '''
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    thickness-=1
                    for cnt in contours:
                        #cnt = contours[4]
                        rbg = tuple(int(colors[i].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        cv.drawContours(im, 
                                        [cnt],
                                        0, 
                                        rbg, 
                                        thickness)
                        cv.putText(im,"mask_"+str(i),(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, thickness,rbg)
                        i=i+1
                    
                    '''
                    cv.imwrite(testFname,im)
                    print("len contours:",len(contours),hierarchy)
                    #return
            VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

    def findAllContours(self, LabelClass,LabelClassName):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                    Regions = []
                    RegionAttributes = []
                    areasInOrderSaved = []
                    
                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    for area in sorted(areas,reverse=True):
                        origIndex = areas.index(area)
                        if len(Regions) <= 2:
                            if area > 15 or len(Regions) == 0:
                                areasInOrderSaved.append(area)
                                cnt = contours[origIndex]                        
                                X = []
                                Y = []
                                epsilon = 0.01*cv.arcLength(cnt,True)
                                approx = cv.approxPolyDP(cnt,epsilon,True)
                                for points in approx:
                                    x =int(points[0][0])
                                    y = int(points[0][1])
                                    X.append(x)
                                    Y.append(y)
                                cv.drawContours(im,[approx],0,rbg,1)
                                cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                                RegionAttributes.append(LabelClassName)
                                Regions.append([X,Y])
                        else: 
                            break
                    print(areasInOrderSaved,"------------",areas)
                            
                    VIA.addFrameMultiRegion(non_pred_name, fileSizeInBytes, Regions, RegionAttributes)
                    cv.imwrite(testFname,im)
                    if False:
                        if len(contours) > 2:
                            print("=======================================================================================================================================================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        elif len(contours) > 1:
                            print("===================================================================================>",LabelClassName)
                            print("len contours:",len(contours),hierarchy)
                        else:
                            print("len contours:",len(contours),hierarchy)
                    #return
            VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

    def findContours(self, LabelClass,LabelClassName):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,LabelClass,Trial)
            OutRoot = TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contours")
            PointsRoot = os.path.join(self.CWD,self.task,"contour_points",LabelClass)
            #TrialRoot.replace("context-prediction-main\\"+self.task,"context-prediction-main\\"+self.task+"\\contour_points")
            VIATemplate =  os.path.join(self.CWD,"contour_template.json")
            VIAOutput =  os.path.join(PointsRoot,Trial+".json")
            # load json points for trial
            VIA = utils.ViaJSONTemplate(VIATemplate)

            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if ".png" not in file:
                        continue
                    
                    imageFname = os.path.join(TrialRoot,file)
                    if(not os.path.isdir(OutRoot)):
                        path = pathlib.Path(OutRoot)
                        path.mkdir(parents=True, exist_ok=True)
                    if(not os.path.isdir(PointsRoot)):
                        path = pathlib.Path(PointsRoot)
                        path.mkdir(parents=True, exist_ok=True)

                    #outFname =  os.path.join(OutRoot,file.replace(".png",".npy"))
                    non_pred_name = file.replace("_pred","")
                    videoFrame = os.path.join(self.imagesDir,Trial,non_pred_name)
                    testFname = os.path.join(OutRoot,file)
                    frameNumber = int(file.replace(".png","").split("_")[1])
                    im = cv.imread(imageFname)
                    fileSizeInBytes = os.path.getsize(videoFrame)
                    imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
                    ret, thresh = cv.threshold(imgray, 1, 255, 0)

                    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST   , cv.CHAIN_APPROX_SIMPLE)
                    #colors = 
                    if(len(contours) ==0):continue
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    
                    for k in range(len(contours)):                    
                        cnt = contours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                    rbg = tuple(int(colors[0].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))        
                    cnt = contours[largestIndex]
                    epsilon = 0.01*cv.arcLength(cnt,True)
                    approx = cv.approxPolyDP(cnt,epsilon,True)
                    cv.drawContours(im,[approx],0,rbg,1)
                    cv.putText(im,LabelClassName,(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX,0.5,rbg)
                    #rbg = tuple(int(colors[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
                    #cv.drawContours(im,approx,0,rbg,thickness)
                    X = []
                    Y = []
                    for points in approx:
                        x =int(points[0][0])
                        y = int(points[0][1])
                        X.append(x)
                        Y.append(y)
                    VIA.addFrame(non_pred_name, fileSizeInBytes, X,Y)
                    #VIA.addRings(file, LabelClassName, PolyPointsY)
                    '''
                    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    thickness-=1
                    for cnt in contours:
                        #cnt = contours[4]
                        rbg = tuple(int(colors[i].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                        cv.drawContours(im, 
                                        [cnt],
                                        0, 
                                        rbg, 
                                        thickness)
                        cv.putText(im,"mask_"+str(i),(cnt[0][0][0],cnt[0][0][1]), cv.FONT_HERSHEY_SIMPLEX, thickness,rbg)
                        i=i+1
                    
                    '''
                    cv.imwrite(testFname,im)
                    print("len contours:",len(contours),hierarchy)
                    #return
            VIA.save(VIAOutput)
            TrialNum+=1
        print("Processed ",TrialNum,"trials")

''

main();