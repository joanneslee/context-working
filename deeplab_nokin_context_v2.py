from email import message
import os, sys
import utils
import pathlib
from scipy import ndimage
import cv2 as cv
from shapely import geometry as geo
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import polygonize
import math

#import metrics
#import geopandas as gpd

global sStarted
global isSRight
isSRight = True
sStarted = False


def main():

    dir=os.getcwd()
    
    task = "Needle_Passing"
    try:
        task=sys.argv[1]
        #GT=sys.argv[1]
        #task=sys.argv[1]
        #print(task)
    except:
        print("Error: no task provided \nUsage: python draw_labels.py <task>")
        #available_tasks = next(os.walk(os.path.join(dir, "images")))[1]
        #print("Available task images: ", available_tasks)
        #available_labels = next(os.walk(os.path.join(dir, "labels")))[1]
        #print("Available task labels: ", available_labels)
        #sys.exit()
    
    I = Iterator(task)
    #I.DrawDeepLab() #TODO: get ctx lines like consensus
    I.GenerateContext(SAVE=True)
    #I = metrics.MetricsIterator(task)
    #I.IOU()
    #I.DrawLabelsContext()
    quit();

class Iterator:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")
        self.cogitoDir = os.path.join(self.CWD, task,"annotations")
        self.cogitoOutputDir = os.path.join(self.CWD, task,"cogito_labeled_images")
        self.context_output = os.path.join(self.CWD,task,"vis_context_labels_v5")        
        self.deeplabOutputDir= os.path.join(self.CWD,task,"deeplab_labeled_images")     
        self.ctxConsensusDir = os.path.join(self.CWD,task,"ctx_consensus")   
        self.ctxSurgeonDir = os.path.join(self.CWD,task,"ctx_surgeon")   
        self.ctxPredDir =  os.path.join(self.CWD,task,"ctx_output")   
        self.tissueDir =  os.path.join(self.CWD,task,"tissue_keypoints")
        self.grasperJawDir = os.path.join(self.CWD,task,"grasper_jaw_keypoints")
        self.contourDir =  os.path.join(self.CWD,task,"contour_points")

        self.OS = "windows" 
        
    def getThreadContours(self,threadMaskImage):
        im = cv.imread(threadMaskImage)
        imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
        ret, thresh = cv.threshold(imgray, 1, 255, 0)
        ThreadContours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) 
        return ThreadContours

    def findClosestIndex(self,frameNumber,tfs):
        for i in range(0,len(tfs)):
            if(frameNumber == tfs[i]):
                return tfs[i]

            if(frameNumber < tfs[i]):
                if i == 0: 
                    return tfs[0]
                else:
                    return tfs[i-1]

            if(frameNumber > tfs[i]):
                if (i+1) >= len(tfs):
                    return  tfs[i]

    def distGraspers(self,P):
        P1,P2 = self.getGrasperBisector(P)  #[(c1[0]+c2[0])/2,(c1[1]+c2[1])/2],[(c3[0]+c4[0])/2,(c3[1]+c4[1])/2] 
        return math.hypot(P1[0]-P2[0],P1[1]-P2[1]),P1[0]-P2[0],P1[1]-P2[1]

    def getGrasperBisector(self,Points):
        c1 = Points[0]
        c2 = Points[1]

        c3 = Points[-2]
        c4 = Points[-1]

        return [(c1[0]+c2[0])/2,(c1[1]+c2[1])/2],[(c3[0]+c4[0])/2,(c3[1]+c4[1])/2] 

    def getTissueBisector(self,Points):
        c1 = Points[0]
        c2 = Points[1]

        c3 = Points[-2]
        c4 = Points[-1]

        return [(c1[0]+c2[0])/2,(c1[1]+c2[1])/2],[(c3[0]+c4[0])/2,(c3[1]+c4[1])/2] 

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def contour_intersect(self,cnt_ref,cnt_query):
    ## Contour is a list of points
    ## Connect each point to the following point to get a line
    ## If any of the lines intersect, then break
        for ref_idx in range(len(cnt_ref)-1):
        ## Create reference line_ref with point AB
            A = cnt_ref[ref_idx][0]
            B = cnt_ref[ref_idx+1][0] 
        
            for query_idx in range(len(cnt_query)-1):
                ## Create query line_query with point CD
                C = cnt_query[query_idx][0]
                D = cnt_query[query_idx+1][0]
            
                ## Check if line intersect
                if self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D):
                    ## If true, break loop earlier
                    return True

        return False

    def GenerateContext(self,SAVE=False):
        count = 0
        Tissues = {} 
        TissueFrames = {}
        GrasperJaws = {}
        GrasperFrames = {}        
        DeeplabVIAPoints = {}
        DeeplabVIAFrames = {}
        DeeplabVIARings = {}

        for root, dirs, files in os.walk(self.tissueDir):
            for file in files:
                fileID = file.replace(".json","") 
                #if "frame" not in file:
                tissueFname = os.path.join(root,file)
                T = utils.ViaJSONInterface(tissueFname)
                Tissues[fileID] = T.getDataDict()
                TissueFrames[fileID] = T.getFrameNumbers()
                print("Input: Tissue Keypoints",file)

        for root, dirs, files in os.walk(self.grasperJawDir):
            for file in files:
                fileID = file.replace(".json","") 
                #if "frame" not in file:
                jawsFname = os.path.join(root,file)      
                J = utils.ViaJSONInterface(jawsFname)
                GrasperJaws[fileID] = J.getDataDict()
                GrasperFrames[fileID] = J.getFrameNumbers()
                print("Input: Grasper Jaws",file)

        for root, dirs, files in os.walk(self.contourDir):
            for file in files:
                fileID = file.replace(".json","") 
                #if "frame" not in file:
                label_class = os.path.basename(root)
                if "ring" in label_class:
                    VIAFname = os.path.join(root,file)
                    J = utils.VIARingJSON(VIAFname)
                    if fileID not in DeeplabVIARings.keys():
                        DeeplabVIARings[fileID] = {}
                    #if fileID not in DeeplabVIAFrames.keys():
                    #    DeeplabVIAFrames[file.replace(".json","")] = {}      
                    ringNames = ["Ring_4","Ring_5","Ring_6","Ring_7"]
                    R4Dict,R5Dict,R6Dict,R7Dict =  J.getRingsPoints()
                    #[R4Frames,R5Frames,R6Frames,R7Frames] = J.getFrameNumbers()                    
                    DeeplabVIARings[fileID]["Ring_4"] = R4Dict
                    DeeplabVIARings[fileID]["Ring_5"] = R5Dict
                    DeeplabVIARings[fileID]["Ring_6"] = R6Dict
                    DeeplabVIARings[fileID]["Ring_7"] = R7Dict
                        #DeeplabVIARings[file.replace(".json","")][ringClass] = J.getFrameNumbers()
                    print("Input: dl_rings",file,label_class)
                else:
                    VIAFname = os.path.join(root,file)
                    J = utils.VIAPolyJSON(VIAFname)
                    class_name = J.getClassKey(label_class)
                    if fileID not in DeeplabVIAPoints.keys():
                        DeeplabVIAPoints[fileID] = {}
                    if fileID not in DeeplabVIAFrames.keys():
                        DeeplabVIAFrames[fileID] = {}
                    DeeplabVIAPoints[fileID][class_name] = J.getDataDict()
                    DeeplabVIAFrames[fileID][class_name] = J.getFrameNumbers()
                    print("Input: dl_points",file,label_class)

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
            #if "Needle_Passing_S05_T03" not in file:
            #    continue
            TrialRoot = os.path.join(self.imagesDir,Trial)
            contextLines = []
            ctxFName = os.path.join(self.ctxConsensusDir, Trial+".txt")  
            ctxPredFName = os.path.join(self.context_output,Trial+".txt")  
            frameNum = 0
            currentRing = 7
            for root, dirs, files in os.walk(TrialRoot):
                for file in files:
                    if "frame" not in file:
                        continue
                    frameNum+=1
                    imageRoot = root 
                    frameNumber = int(file.replace(".png","").split("_")[1])        
                    #if( frameNumber< 5628):continue            
                    trialFname = os.path.basename(root)
                    imageFName = os.path.join(imageRoot, file)
                    cogitoRoot = root.replace("images","annotations")                
                    cogitoFName = os.path.join(cogitoRoot, utils.imageToJSON(file))
                    outputRoot =  os.path.join(self.deeplabOutputDir,trialFname)
                    outputFName = os.path.join(self.deeplabOutputDir,trialFname, file)
                    #print("outputFName",outputFName)
                    closestIndex = -1
                    if "Suturing" in self.task: 
                        TissueClosestIndex = TissueFrames[trialFname][0]
                        #frameNumber = 109
                        tfs = TissueFrames[trialFname]
                        TissueClosestIndex = self.findClosestIndex(frameNumber,tfs)                    
                        TissuePoints = Tissues[trialFname][str(TissueClosestIndex)]
                        GrasperJawPoints = GrasperJaws[trialFname][str(TissueClosestIndex)]
                        closestIndex = TissueClosestIndex
                        #print("Drawing Segmentation labels and Keypoints:", os.path.basename(root),file)
                    else:
                        gfs = GrasperFrames[trialFname]
                        GrasperClosestIndex = self.findClosestIndex(frameNumber,gfs)              
                        GrasperJawPoints = GrasperJaws[trialFname][str(GrasperClosestIndex)]
                        closestIndex = GrasperClosestIndex
                    
                    All_dl_points = DeeplabVIAPoints[trialFname] #["class"]["frame"] = list of points
                    if "Needle" in self.task:
                        All_ring_points = DeeplabVIARings[trialFname]
                        try:
                            Ring4Points = All_ring_points["Ring_4"][str(frameNumber)]
                            Ring5Points = All_ring_points["Ring_5"][str(frameNumber)]
                            Ring6Points = All_ring_points["Ring_6"][str(frameNumber)]
                            Ring7Points = All_ring_points["Ring_7"][str(frameNumber)]
                        except Exception as e:
                            print(e,"in RingPoints")
                            Ring4Points = {}
                            Ring5Points = {}
                            Ring6Points = {}
                            Ring7Points = {}

                    RgrasperRoot = root.replace("images","deeplab_grasper_R_v3")
                    LgrasperRoot = root.replace("images","deeplab_grasper_L_v3")
                    threadRoot = root.replace("images","deeplab_thread_v3")
                    ringRoot = root.replace("images","deeplab_rings_v3")
                    needleRoot = root.replace("images","deeplab_needle_v3")
                    annotationRoot = root.replace("images","annotations")

                    RgrasperMask = os.path.join(RgrasperRoot, utils.imageToNPY(file))
                    LgrasperMask = os.path.join(LgrasperRoot, utils.imageToNPY(file))
                    threadMask = os.path.join(threadRoot, utils.imageToNPY(file))
                    threadMaskImage = os.path.join(threadRoot, file.replace(".png","_pred.png"))
                    needleMask = os.path.join(needleRoot, utils.imageToNPY(file))
                    ringMask = os.path.join(ringRoot, utils.imageToNPY(file))
                    annotationFile = os.path.join(annotationRoot, utils.imageToJSON(file))
                    try:
                        LgrasperPoints = All_dl_points["dl_grasper_L"][str(frameNumber)]
                    except Exception as e:
                        print(e,"in LgrasperPoints")
                        LgrasperPoints = {}

                    try:
                        RgrasperPoints = All_dl_points["dl_grasper_R"][str(frameNumber)]
                    except Exception as e:
                        print(e,"in RgrasperPoints")
                        RgrasperPoints = {}
                    if "Needle" in self.task or "Suturing" in self.task:
                        try:                        
                            NeedlePoints = All_dl_points["dl_needle"][str(frameNumber)]
                        except Exception as e:
                            print(e,"in NeedlePoints")
                            NeedlePoints = {}

                    

                    #L_grasper = utils.NPYInterface3.loadArr(LgrasperMask)
                    #R_grasper = utils.NPYInterface3.loadArr(RgrasperMask)
                    #Thread = utils.NPYInterface3.loadArr(threadMask)
                    #if "Suturing" in self.task or "Needle" in self.task: Needle = utils.NPYInterface3.loadArr(needleMask)                    
                    #if "Needle" in self.task: Rings = utils.NPYInterface3.loadArr(ringMask)
                        
                    #MPI = MPInterface(MP_comb) # turn on for MPs as well

                    CtxI = utils.ContextInterface2(ctxFName)
                    CtxI_Pred = utils.ContextInterface(ctxFName,True)
                    J = utils.JSONInterface(annotationFile)

                    gtPolygons = J.getPolygonsDict(); # graspers only in KT, 
                    gtKeypoints = J.getKeyPointsDict(); # None in KT,
                    cn,polylineSeries = J.getPolyLines();
                    SingleThreadX = []
                    SingleThreadY = []
                    for i in range(len(polylineSeries)):
                        l = len(polylineSeries)
                        for j in range(0,len(polylineSeries[i]),2):
                            SingleThreadX.append(polylineSeries[i][j])
                            SingleThreadY.append(polylineSeries[i][j+1])
                    SingleThreadPoints = [(SingleThreadX[i],SingleThreadY[i]) for i in range(len(SingleThreadX))]

                    TContours = self.getThreadContours(threadMaskImage)
                    ThreadContours = []
                    minArea = 80
                    for k in range(len(TContours)):                    
                        cnt = TContours[k]
                        area = cv.contourArea(cnt)
                        if area >= minArea:
                            ThreadContours.append(cnt)
                    #print("Areas",areas)

                    try:
                        L_Dist = utils.distTwoPoints(GrasperJawPoints[2],GrasperJawPoints[3])
                        R_Dist = utils.distTwoPoints(GrasperJawPoints[0],GrasperJawPoints[1])
                        Grasper_Dist,Grasper_DistX,Grasper_DistY = self.distGraspers(GrasperJawPoints)
                    except Exception as e:
                        L_Dist = 0
                        R_Dist = 0 
                    PARAM_JAW_DIST = 25
                    #from orig 12, NP is best at 18
                    #NP 26: 0.5641
                    #NP 42: 0.57910839234271
                    if(L_Dist < PARAM_JAW_DIST):
                        L_Gripping = True
                    else: 
                        L_Gripping = False
                    if(R_Dist < PARAM_JAW_DIST):
                        R_Gripping = True
                    else:
                        R_Gripping = False

                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)
                        
                    pred, gt = self.GetGenShapes(gtPolygons,gtKeypoints,SingleThreadPoints,polylineSeries,ThreadContours,LgrasperPoints,RgrasperPoints)

                    # maskify Thread
                    if("Knot" in self.task):
                        #pred, gt = self.GetKTShapes(gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints)     
                        
                        #dev                    
                        ctxPredLine, LG_inter_T, RG_inter_T = self.GenerateContextLineKT(pred, gt ,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX)    

                        #GT                    
                        #ctxPredLine, LG_inter_T, RG_inter_T = self.GenerateContextLineKT(gt,pred, L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX)                         
                        
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        #self.DrawSingleImageContextKT(pred, gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T)
                    if("Needle" in self.task):
                        #pred, gt = self.GetGenShapes(gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints)
                        ringShapes, ringShapes_gt = self.GetRingShapes(Ring4Points,Ring5Points,Ring6Points,Ring7Points,gtPolygons)
                        needleShape, needleShape_gt = self.GetNeedleShapes(NeedlePoints,gtPolygons)
                        #gt, pred, ringShapes_gt, ringShapes, needleShape_gt, needleShape -> GT Test
                        #ctxPredLine, LG_inter_T, RG_inter_T,messages = "",0,0,[]                    # this causes list error, switching gt with pred
                        
                        #pred, gt, ringShapes, ringShapes_gt, needleShape, needleShape_gt -> dev
                        ctxPredLine, LG_inter_T, RG_inter_T,messages = self.GenerateContextLineNP(pred, gt, ringShapes, ringShapes_gt, needleShape, needleShape_gt, L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX,currentRing)                        
                        #gt, pred, ringShapes_gt, ringShapes, needleShape_gt, needleShape -> GT Test 
                        #ctxPredLine, LG_inter_T, RG_inter_T,messages = self.GenerateContextLineNP(gt, pred, ringShapes_gt, ringShapes, needleShape_gt, needleShape, L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX,currentRing)

                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        #self.DrawSingleImageContextNP(pred, gt, ringShapes, ringShapes_gt, needleShape, needleShape_gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T,messages,GT=False)
                    if("Suturing" in self.task):              
                        needleShape, needleShape_gt = self.GetNeedleShapes(NeedlePoints,gtPolygons)
                        
                        gt_bisector, gt_tissue, pred_bisector, pred_tissue,Bisector = self.GetTissueDist(TissuePoints,needleShape, needleShape_gt) 

                        #dev
                        ctxPredLine, LG_inter_T, RG_inter_T,messages = self.GenerateContextLineS(pred, gt ,needleShape,needleShape_gt ,L_Gripping,R_Gripping,frameNumber,contextLines,gt_bisector, gt_tissue, pred_bisector, pred_tissue,Bisector)
                        
                        #gt
                        #ctxPredLine, LG_inter_T, RG_inter_T,messages = self.GenerateContextLineS(gt, pred, needleShape_gt, needleShape, L_Gripping,R_Gripping,frameNumber,contextLines,gt_bisector, gt_tissue, pred_bisector, pred_tissue,Bisector)
                                                
                        messages.append("LJ Dist:"+"{:.2f}".format(L_Dist) + str(L_Gripping))
                        messages.append("RJ Dist:"+"{:.2f}".format(R_Dist)+ str(R_Gripping))
                        contextLines.append(ctxPredLine)
                        #print(Trial,frameNumber,ctxPredLine)
                        #self.DrawSingleImageContextS(pred, gt,needleShape,needleShape_gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T,messages)
                    count += 1
            print("proc",os.path.basename(TrialRoot),"count:",frameNum)
            if(len(contextLines) > 2 and SAVE):                
                print("saving",ctxPredFName)
                count+=1
                outdir = os.path.abspath(ctxPredFName + "/../")
                if(not os.path.isdir(outdir)):
                    path = pathlib.Path(outdir)
                    path.mkdir(parents=True, exist_ok=True)     
                utils.save(ctxPredFName,contextLines)             
        print(count,"images processed!")

    def GetTissueDist(self,TissuePoints,needleShape, needleShape_gt):
        gt_bisector, gt_tissue, pred_bisector, pred_tissue = 1000,1000,1000,1000
        #if not isinstance(needleShape,list):
        #    for n in needleShape.geoms:
        P1,P2 = self.getTissueBisector(TissuePoints)
        p1=np.array(P1)
        p2=np.array(P2)
        p3=[(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
        Bisector = geo.LineString([p1,p2,p3])

        if not isinstance(needleShape,list):
            for n in needleShape.geoms:
                d = n.distance(Bisector)
                if d < pred_bisector:
                    pred_bisector = d

        if not isinstance(needleShape_gt,list):
            for n in needleShape_gt.geoms:
                d = n.distance(Bisector)
                if d < gt_bisector:
                    gt_bisector = d
        
        #min_needle_center_dist_point = np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))

        for tp in TissuePoints:
            P = geo.Point(tp)
            if not isinstance(needleShape,list):
                for n in needleShape.geoms:
                    d = n.distance(P)
                    if d < pred_tissue:
                        pred_tissue = d

            if not isinstance(needleShape_gt,list):
                for n in needleShape_gt.geoms:
                    d = n.distance(P)
                    if d < gt_tissue:
                        gt_tissue = d

        return gt_bisector, gt_tissue, pred_bisector, pred_tissue,Bisector
            

    def GenerateContextLineS(self,pred, gt,needleShape,needleShape_gt ,L_Gripping,R_Gripping,frameNumber,contextLines,gt_bisector, gt_tissue, pred_bisector, pred_tissue,Bisector, GT=False):
        
        def last5thState(s):
            return s.split(" ")[-1]   
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt 
        # if pred_bisector, pred_tissue are both 0 -> needle on top of the canvas
        messages = []
        Faulty = False
        if GT:
            RG_inter_T = 0
            #RingDistances_L = [ min([LG_dl.distance(shape) for shape in R_GROUP ])  for R_GROUP in ringShapes_gt]
            #RingDistances_R = [ min([RG_dl.distance(shape) for shape in R_GROUP ])  for R_GROUP in ringShapes_gt]
            NeedleDistances_L =  min([LG_dl.distance(shape) for shape in needleShape_gt if not isinstance(needleShape_gt,list)] )
            NeedleDistances_R =  min([RG_dl.distance(shape) for shape in needleShape_gt if not isinstance(needleShape_gt,list)] )
        else:
            if not isinstance(LG_dl,list) :
                #if len(LG_dl.geoms) > 0:
                LG_dist_T = LG_dl.distance(T_dl)
                if(len(LG_dl.geoms))>0:
                    #LG_x_center,LG_y_center = LG_dl.geoms[0].centroid.x,LG_dl.geoms[0].centroid.y     
                    pass
                else:
                    #LG_x_center,LG_y_center = 400,200
                    pass
            else: 
                LG_dist_T = 100

            if not isinstance(RG_dl,list) :
                #if len(LG_dl.geoms) > 0:
                RG_dist_T = RG_dl.distance(T_dl)
            else: 
                RG_dist_T = 100
            
           
           
            try:
                if not isinstance(needleShape,list):
                    LG_dist_N =  min([LG_dl.distance(shape) for shape in needleShape.geoms if not isinstance(needleShape,list)] )
                    RG_dist_N =  min([RG_dl.distance(shape) for shape in needleShape.geoms if not isinstance(needleShape,list)] )

                    RG_inter_N =  max([LG_dl.intersection(shape).area for shape in needleShape.geoms if not isinstance(needleShape,list)] )
                    LG_inter_N =  max( [RG_dl.intersection(shape).area for shape in needleShape.geoms if not isinstance(needleShape,list)] ) 
                    N_inter_TS = pred_tissue < 2 
                    messages.append("Min LN"+"{:.2f}".format(LG_dist_N))
                    messages.append("Min RN"+"{:.2f}".format(RG_dist_N))

                    messages.append("Inter LN"+"{:.2f}".format(RG_inter_N))
                    messages.append("Inter RN"+"{:.2f}".format(LG_inter_N))
                    messages.append("N to Tissue"+"{:.2f}".format(pred_tissue))
                    messages.append("N to Bisector"+"{:.2f}".format(pred_bisector))

                    bi_x,bi_y = Bisector.centroid.x,Bisector.centroid.y
                    n_x,n_y = needleShape.geoms[0].centroid.x,needleShape.geoms[0].centroid.y
                else:
                    Faulty = True
                #RingDistances_N = [ min([needleShape.distance(shape) for shape in R_GROUP.geoms if not isinstance(needleShape,list) ])  for R_GROUP in ringShapes if not isinstance(R_GROUP,list)] if ringShapes != [] else []
                #RingInter_N =  [ max([needleShape.intersection(shape).area for shape in R_GROUP.geoms if not isinstance(needleShape,list)])  for R_GROUP in ringShapes  if not isinstance(R_GROUP,list)] if ringShapes != [] else []
                
            except Exception as e:
                print(e,"could not load a shape in GenerateContextLineS")
                messages.append("missing contour")
                Faulty = True

        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        Extra_State = 0
        INTER_THRESH = 1
        global sStarted
        global isSRight

        if not Faulty:
            if(R_Gripping):                
                if(RG_dist_N < INTER_THRESH ):
                    R_G_Hold = 2
                elif(RG_dist_T < INTER_THRESH):
                    R_G_Hold = 3
            else: #Right not gripping
                if(RG_dist_N < INTER_THRESH):
                    R_G_Touch = 2
                elif(RG_dist_T < INTER_THRESH):
                    R_G_Touch = 3

            if(L_Gripping):            
                if(LG_dist_N < INTER_THRESH):
                    L_G_Hold = 2
                elif(LG_dist_T < INTER_THRESH):
                    L_G_Hold = 3
            else:#Left not gripping
                if(LG_dist_N < INTER_THRESH):
                    L_G_Touch = 2
                elif(LG_dist_T < INTER_THRESH):
                    L_G_Touch = 3

            if not sStarted:
                Extra_State = 2
                if L_G_Touch == 2:
                    sStarted = True
                if LG_dist_N < 200:
                    sStarted = True
            if sStarted:
                if len(contextLines) > 0:
                    last = last5thState(contextLines[-1])
                else:
                    last = "0"
            
                if(bi_x > n_x):
                    isSRight = False
                else:
                    isSRight = True

                if(last == "0"):
                    if(pred_tissue < 10 and pred_bisector>7 and isSRight):
                        Extra_State = 1
                elif(last == "1"):
                    if not isSRight:
                        if(pred_tissue < 9):
                            Extra_State = 1
                        else:
                            Extra_State = 0
                    else:
                        if(pred_tissue < 10):
                            Extra_State = 1
                        else:
                            Extra_State = 2

                elif(last == "2"):
                    if not isSRight:
                        if not R_Gripping:
                            Extra_State = 0
                        else:
                            Extra_State = 2
                    else:
                        if RG_dist_N > 40 or not R_Gripping:
                            Extra_State = 0
                        else:
                            Extra_State = 2

            #if(R_G_Hold == 2):
            #    Extra_State = 1 #if needle end is near the tissue
            #if(L_G_Hold == 2):
            #    Extra_State = 1 #if needle end is near the tissue  
            #if(R_G_Hold == 2 and L_G_Hold == 2 and pred_tissue < 5 and pred_bisector > 20):
            #    Extra_State = 2

        elif (len(contextLines) > 0):
            s_ = contextLines[-1].split(" ")
            L_G_Hold =s_[1]
            L_G_Touch = s_[2]
            R_G_Hold = s_[3]
            R_G_Touch = s_[4]
        else:
            L_G_Hold = 0
            L_G_Touch =0
            R_G_Hold = 0
            R_G_Touch =0
            Extra_State=2
        
        messages.append("1:"+str(sStarted))        
        messages.append("2:"+str(isSRight))
        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State), LG_dist_T,RG_dist_T,messages
       
    def GenerateContextLineNP(self,pred, gt, ringShapes,ringShapes_gt,needleShape,needleShape_gt ,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX,currentRing, GT=False):
        np1 = [184,287,398,496]
        np1x = [235,342,447]
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt  
        [R4_Group,R5_Group,R6_Group,R7_Group] = ringShapes
        [R4_Group_gt,R5_Group_gt,R6_Group_gt,R7_Group_gt] = ringShapes_gt
        messages = []
        Faulty = False
        if GT:
            RG_inter_T = 0
            RingDistances_L = [ min([LG_dl.distance(shape) for shape in R_GROUP ])  for R_GROUP in ringShapes_gt]
            RingDistances_R = [ min([RG_dl.distance(shape) for shape in R_GROUP ])  for R_GROUP in ringShapes_gt]
        else:            
            LG_inter_T = LG_dl.distance(T_dl) if not isinstance(LG_dl,list) else 20
            RG_inter_T = RG_dl.distance(T_dl) if not isinstance(RG_dl,list) else 20   

            if not isinstance(LG_dl,list) :
                if len(LG_dl.geoms) > 0:
                    LG_x_center,LG_y_center = LG_dl.geoms[0].centroid.x,LG_dl.geoms[0].centroid.y     
            else:
                LG_x_center,LG_y_center = 400,200

            try:
                RingDistances_L = [ min([LG_dl.distance(shape) for shape in R_GROUP.geoms if not isinstance(LG_dl,list)])  for R_GROUP in ringShapes if not isinstance(R_GROUP,list) ] if ringShapes != [] else []

                RingDistances_R = [ min([RG_dl.distance(shape) for shape in R_GROUP.geoms if not isinstance(RG_dl,list) ])  for R_GROUP in ringShapes if not isinstance(R_GROUP,list) ] if ringShapes != [] else []  
                RingDistances_N = []
                absMinDistN = 10000
                ringID = -1
                for i in range(len(ringShapes)):
                    R_GROUP = ringShapes[i]
                    if not isinstance(R_GROUP,list):
                        currMin = min([needleShape.distance(shape) for shape in R_GROUP.geoms ])
                        RingDistances_N.append(currMin)
                        if currMin < absMinDistN:
                            ringID = i
                            absMinDistN = currMin
                #closestRingCenterX,closestRingCenterY = ringShapes[ringID].geoms[0].centroid.x,ringShapes[ringID].geoms[0].centroid.x,
                closestRingCenterX,closestRingCenterY = ringShapes[ringID].centroid.x,ringShapes[ringID].centroid.y
                
                RingDistances_N = [ min([needleShape.distance(shape) for shape in R_GROUP.geoms if not isinstance(needleShape,list) ])  for R_GROUP in ringShapes if not isinstance(R_GROUP,list)] if ringShapes != [] else []
                RingInter_N =  [ max([needleShape.intersection(shape).area for shape in R_GROUP.geoms if not isinstance(needleShape,list)])  for R_GROUP in ringShapes  if not isinstance(R_GROUP,list)] if ringShapes != [] else []
                minLRing = min(RingDistances_L)
                minRRing = min(RingDistances_R)
                minNRing = min(RingDistances_N)
                maxInter_RN = max(RingInter_N)
                messages.append("Min L Ring:"+"{:.2f}".format(minLRing))
                messages.append("Min R Ring:"+"{:.2f}".format(minRRing))
                messages.append("Min N Ring:"+"{:.2f}".format(minNRing))
                messages.append("Max RN Inter:"+"{:.2f}".format(maxInter_RN))
                x_center,y_center = needleShape.centroid.x,needleShape.centroid.y


                dists = [ abs(x_center-p) for p in np1]
                dists2 = [ abs(x_center-p) for p in np1x]
                DistanceToRingCenter = min(dists)
                DistanceToRingInCenter = min(dists2)
                messages.append("ToRing:"+"{:.2f}".format(DistanceToRingCenter))
                messages.append("ToVoid:"+"{:.2f}".format(DistanceToRingInCenter))
                
            except Exception as e:
                print(e,"could not load a shape in GenerateContextLineNP")
                messages.append("missing contour")
                Faulty = True


        def last5thState(s):
            return s.split(" ")[-1]            
        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        Extra_State = 0
        INTER_THRESH = 1
        #geo.Polygon().centroid
        #(y_nearby_ring,x_nearby_ring) = ndimage.center_of_mass(LocalRings)
        #localRingsMass = np.sum(LocalRings)        
        
        #print("Needle Center",y_center,x_center )
        #print("needle y_center:",y_center,"x_center:",x_center)

        #dists = [ abs(x_center-p) for p in np1]
        #dists2 = [ abs(x_center-p) for p in np1x]
        #distsROI1 = [ abs(x_center-p) for p in np1]        
        #distsROI2 = [ abs(x_center-p) for p in np1x]
        #d = min(dists)
        #d2 = min(dists2)
        #ROIXmin = min(distsROI2)
        #print("dists 1:",d,":",dists)
        #print("dists 2:",d2,":",dists2)

        #dists2 = [ abs(x_center-p) for p in np1x]
        
        if not Faulty:
            #print("\t===>Ignoring Needle")
            if(R_Gripping):
                if(minRRing < INTER_THRESH):
                    R_G_Hold = 2
                elif(RG_inter_T <INTER_THRESH ):
                    R_G_Hold = 3
            else: #Right not gripping
                if(minRRing <INTER_THRESH):
                    R_G_Touch = 2
                elif(RG_inter_T <INTER_THRESH):
                    R_G_Touch = 3
            if(L_Gripping):            
                if(minLRing  <INTER_THRESH):
                    L_G_Hold = 2
                elif(LG_inter_T <INTER_THRESH):
                    L_G_Hold = 3
            else:#Left not gripping
                if(minLRing  <INTER_THRESH):
                    L_G_Touch = 2
                elif(LG_inter_T  <INTER_THRESH):
                    L_G_Touch = 3
        
        elif (len(contextLines) > 0):
            s_ = contextLines[-1].split(" ")            
            L_G_Hold =s_[1]
            L_G_Touch = s_[2]
            R_G_Hold = s_[3]
            R_G_Touch = s_[4]
        else:
            L_G_Hold = 0
            L_G_Touch =0
            R_G_Hold = 0
            R_G_Touch =0
        
        if(len(contextLines) == 0):
            Extra_State = 0
        else:
            #print("\tNeedle sum, LocalRing sum",needleSum,localRingsMass,"accepted 5th?",str( needleSum > 50 and localRingsMass >20))
            #print("")
            if not Faulty:

                last = last5thState(contextLines[-1])
                if last == "0":
                    if minNRing < 10:
                        Extra_State = 1
                        messages.append(str(last)+":minNRing < 10:"+str(Extra_State))
                    else:
                        Extra_State = 0
                        messages.append(str(last)+":minNRing < 10E:"+str(Extra_State))
                elif last == "1":

                    if maxInter_RN > 3:
                        Extra_State = 2
                        messages.append(str(last)+":maxInter_RN > 3:"+str(Extra_State))
                    elif minNRing > 10:
                        Extra_State = 0
                        messages.append(str(last)+":minNRing > 10:"+str(Extra_State))
                    elif (x_center < closestRingCenterX):
                        Extra_State = 8
                        messages.append(str(last)+":x_center < closestRingCenterX:"+str(Extra_State))
                    else:
                        Extra_State = 1
                        messages.append(str(last)+":E:"+str(Extra_State))
                    '''
                    if N_inter_R > 200:
                        Extra_State = 2
                    elif x_center < x_nearby_ring:
                        Extra_State = 2
                    elif not R_Gripping or not L_Gripping:
                        Extra_State = 0
                    else:
                        Extra_State = 1
                    '''
                elif last == "2":
                    if DistanceToRingInCenter < 10:
                        Extra_State = 0
                    elif not R_Gripping or not L_Gripping:
                        Extra_State = 0
                        messages.append(str(last)+":not R_Gripping or not L_Gripping:"+str(Extra_State))
                    elif minNRing > 10:
                        Extra_State = 0
                        messages.append(str(last)+":minNRing > 10:"+str(Extra_State))
                    elif maxInter_RN == 0:
                        Extra_State = 1
                        messages.append(str(last)+":maxInter_RN == 0:"+str(Extra_State))
                    else: 
                        Extra_State = 2
                        messages.append(str(last)+":E:"+str(Extra_State))
             
            else:
                last = last5thState(contextLines[-1])
                '''
                
                if last == "0":                    
                    if (x_center < closestRingCenterX ):
                        Extra_State = 2
                    elif minNRing < 10:
                        Extra_State = 0
                    else:
                        Extra_State = 1

                if last == "1":                    
                    if (x_center < closestRingCenterX ):
                        Extra_State = 2
                    elif minNRing < 10:
                        Extra_State = 0
                    else:
                        Extra_State = 1

                elif last == "2":
                    if not R_Gripping:
                        Extra_State = 0
                    elif LG_x_center < 20:
                        Extra_State = 0
                    else: 
                        Extra_State = 2
                '''
                Extra_State = last5thState(contextLines[-1])
                messages.append(str(last)+":Faulty:"+str(Extra_State))

            if not Faulty and maxInter_RN > 20:
                Extra_State = 2
                messages.append(str(last)+":not Faulty and maxInter_RN > 20:"+str(Extra_State))
            
        #if(R_G_Hold == 2 and L_G_Hold == 2 and min_Tissue_Dist < 5 and needle_center_dist > 20):
        #    Extra_State = 2

        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str(L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State), LG_inter_T, RG_inter_T, messages 

    def GenerateContextLineNP2(self,inter_couts,L_Gripping,R_Gripping,frameNumber,contextLines,Needle,LocalRings,needleSum,ROI_x, ROI_y):
        np1 = [184,287,398,496]
        np1x = [235,342,447]

        LG_inter_T = 0
        RG_inter_T = 0
        LG_inter_N = 0
        RG_inter_N = 0
        N_inter_R = 0
        
        def last5thState(s):
            return s.split(" ")[-1]
        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        Extra_State = 0
        INTER_THRESH = 120 
        (y_nearby_ring,x_nearby_ring) = ndimage.center_of_mass(LocalRings)
        localRingsMass = np.sum(LocalRings)
        (y_center,x_center) = ndimage.center_of_mass(Needle)
        print("needle y_center:",y_center,"x_center:",x_center)

        dists = [ abs(x_center-p) for p in np1]
        dists2 = [ abs(x_center-p) for p in np1x]
        distsROI1 = [ abs(x_center-p) for p in np1]        
        distsROI2 = [ abs(x_center-p) for p in np1x]
        d = min(dists)
        d2 = min(dists2)
        ROIXmin = min(distsROI2)
        print("dists 1:",d,":",dists)
        print("dists 2:",d2,":",dists2)

        dists2 = [ abs(x_center-p) for p in np1x]

        if needleSum > 25:
            print("\t===>Ignoring Needle")
            if(R_Gripping):
                if(RG_inter_N > INTER_THRESH):
                    R_G_Hold = 2
                elif(RG_inter_T >INTER_THRESH ):
                    R_G_Hold = 3
            else: #Right not gripping
                if(RG_inter_N >INTER_THRESH):
                    R_G_Touch = 2
                elif(RG_inter_T >INTER_THRESH):
                    R_G_Touch = 3
            if(L_Gripping):            
                if(LG_inter_N  >INTER_THRESH):
                    L_G_Hold = 2
                elif(LG_inter_T  >INTER_THRESH):
                    L_G_Hold = 3
            else:#Left not gripping
                if(LG_inter_N  >INTER_THRESH):
                    L_G_Touch = 2
                elif(LG_inter_T  >INTER_THRESH):
                    L_G_Touch = 3
        else:
            s_ = contextLines[-1].split(" ")
            L_G_Hold =s_[1]
            L_G_Touch = s_[2]
            R_G_Hold = s_[3]
            R_G_Touch = s_[4]

        if(len(contextLines) == 0):
            Extra_State = 0
        else:
            print("\tNeedle sum, LocalRing sum",needleSum,localRingsMass,"accepted 5th?",str( needleSum > 50 and localRingsMass >20))
            print("")
            if needleSum > 50 and localRingsMass >8 :
                last = last5thState(contextLines[-1])
                if last == "0":
                    if N_inter_R > 150:
                        Extra_State = 1
                    else:
                        Extra_State = 0
                elif last == "1":                    
                    if (d < d2 ):
                        Extra_State = 2
                        print("\td < d2 ,Extra_State = 2")
                    elif d2 < 10:
                        Extra_State = 0
                    else:
                        Extra_State = 1
                    '''
                    if N_inter_R > 200:
                        Extra_State = 2
                    elif x_center < x_nearby_ring:
                        Extra_State = 2
                    elif not R_Gripping or not L_Gripping:
                        Extra_State = 0
                    else:
                        Extra_State = 1
                    '''
                elif last == "2":
                    if not R_Gripping or not L_Gripping:
                        Extra_State = 0
                    elif d2 < 10:
                        Extra_State = 0
                    else: 
                        Extra_State = 2
                
            else:
                last = last5thState(contextLines[-1])
                if last == "1":                    
                    if (d < d2 ):
                        Extra_State = 2
                    elif d2 < 10:
                        Extra_State = 0
                    else:
                        Extra_State = 1

                elif last == "2":
                    if not R_Gripping:
                        Extra_State = 0
                    elif ROIXmin < 20:
                        Extra_State = 0
                    else: 
                        Extra_State = 2
                
                Extra_State = last5thState(contextLines[-1])

            if N_inter_R > 100:
                Extra_State = 2
            
        #if(R_G_Hold == 2 and L_G_Hold == 2 and min_Tissue_Dist < 5 and needle_center_dist > 20):
        #    Extra_State = 2

        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str( L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State)

    def GenerateContextLineKT(self,pred, gt ,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX, GT=False):
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt  
        if GT:
            RG_inter_T = 0
        else:

            #RG_inter_T = RG_dl.distance(T_dl)
            #LG_inter_T = LG_dl.distance(T_dl)

            if not isinstance(RG_dl,list) :
                RG_inter_T = RG_dl.distance(T_dl)
            else:
                RG_inter_T = 100

            if not isinstance(LG_dl,list) :
                LG_inter_T = LG_dl.distance(T_dl)
            else:
                LG_inter_T = 100

            
            #P1 = LG_dl.centroid
            #P2 = RG_dl.centroid
            #Grasper_DistX = P1.distance(P2)

        def last5thState(s):
            return s.split(" ")[-1]
        L_G_Touch = 0
        L_G_Hold = 0
        R_G_Touch = 0
        R_G_Hold = 0
        Extra_State = 0
        INTER_THRESH = 2.5
        
        if(R_Gripping):
            if(RG_inter_T < INTER_THRESH ):
                R_G_Hold = 3
        else: #Right not gripping
            if(RG_inter_T < INTER_THRESH):
                R_G_Touch = 3
        if(L_Gripping):
            if(LG_inter_T  < INTER_THRESH):
                L_G_Hold = 3
        else:#Left not gripping
            if(LG_inter_T  < INTER_THRESH):
                L_G_Touch = 3
        if(len(contextLines) == 0):
            Extra_State = 0
        else:
            last = last5thState(contextLines[-1])
            if last == "0" and Grasper_DistX < 0:                
                Extra_State = 1 #+wrapped
            elif last == "1":
                if Grasper_DistX > 130:                    
                    Extra_State = 2 # lose
                else:
                    Extra_State = 1
            elif last == "2":
                if Grasper_DistX > 150:
                    Extra_State = 3 # tight
                else:
                    Extra_State = 2 # lose
            elif last == "3":
                if not L_Gripping or not R_Gripping:
                    Extra_State = 0 # tight
                else:
                    Extra_State = 3 # lose 
        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str(L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State), LG_inter_T, RG_inter_T

    def DrawSingleImageContextNP(self,pred, gt,ringShapes,ringShapes_gt,needleShape,needleShape_gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T,messages,GT=False):
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt
        image = cv.imread(imageFName)
        plt.imshow(image, cmap='gray') 
        if GT:        
            try:
                x,y = unary_union(LG_Group_gt).exterior.xy
                plt.plot(x,y)                
                #plt.plot(LG_Group_gt.centroid.x,LG_Group_gt.centroid.y,)
            except Exception as e:
                print(e,"No LG_Group_gt")

            try:
                x,y = unary_union(RG_Group_gt).exterior.xy
                #plt.plot(RG_Group_gt.centroid.x,RG_Group_gt.centroid.y)
                plt.plot(x,y)
            except Exception as e:
                print(e, "no RG GT annotation")
            try:
                cords = T_Group_gt.coords if len(T_Group_gt) > 1 else [0,0]
                #plt.plot(cords)
            except Exception as e:
                print(e,"probably no thread GT annotation")
    


        try:
            for s in LG_dl.geoms:
                x,y = s.exterior.xy
                plt.plot(x,y,color='blue',linewidth=1)
            #x,y = unary_union(LG_dl).exterior.xy
            #plt.plot(x,y)                
        except Exception as e:
            print(e,"No LG DL label")
        try:
            for s in RG_dl.geoms:
                x,y = s.exterior.xy
                plt.plot(x,y,color='blue',linewidth=1)
            #x,y = unary_union(RG_dl).exterior.xy
            #plt.plot(x,y)
        except Exception as e:
            print(e,"No RG DL label") 
        
        try:
            for s in needleShape.geoms:
                x,y = s.exterior.xy
                plt.plot(x,y,color='red',linewidth=1)
            #x,y = unary_union(RG_dl).exterior.xy
            #plt.plot(x,y)
        except Exception as e:
            print(e,"No N DL label") 
        '''
        for ns in needleShape:           

            try:
                x,y = unary_union(ns).exterior.xy
                plt.plot(ns.centroid.x,ns.centroid.y)
                plt.plot(x,y)
            except Exception as e:
                print(e,"No Needle DL label")
        '''

        for ringGoup in ringShapes:
            for rs in ringGoup:
                #if not isinstance(rs,list):
                try:
                    x,y = unary_union(rs).exterior.xy
                    #plt.plot(rs.centroid.x,rs.centroid.y)
                    #plt.circle((rs.centroid.x,rs.centroid.y),1,color='g')
                    plt.plot( rs.centroid.x,rs.centroid.y, ls=' ', marker='o', markersize=5, color='r' )
                    plt.plot(x,y)
                except Exception as e:
                    print(e,"No Ring label")

        
        #unaryThread = unary_union(T_dl.geoms)
        #print("thread geoms:",T_dl.geoms,"unaryThread",unaryThread.is_valid,type(unaryThread))
        for thread in T_dl.geoms:
            if thread.exterior.xy:
                #print(len(t))
                x,y = thread.exterior.xy
                #print(thread.exterior.xy)
                plt.plot(x,y)

        strArr = [CtxI.getContext(frameNumber),ctxPredLine,"LG->T:"+str(LG_inter_T),"RG->T:"+str(RG_inter_T)]
        offset = 1
        for s in strArr:
            x = 10
            y = 22 * offset; 
            plt.text(x,y,s,fontsize=12,color='red')
            offset+=1

        for s in messages:
            x = 10
            y = 22 * offset; 
            plt.text(x,y,s,fontsize=12,color='red')
            offset+=1

        plt.savefig(outputFName)
        plt.close()

    def DrawSingleImageContextS(self,pred, gt,needleShape,needleShape_gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T,messages,GT=False):
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt
        image = cv.imread(imageFName)
        plt.imshow(image, cmap='gray') 
        if GT:        
            try:
                x,y = unary_union(LG_Group_gt).exterior.xy
                plt.plot(x,y)                
                plt.plot(LG_Group_gt.centroid.x,LG_Group_gt.centroid.y)
            except Exception as e:
                print(e,"No LG_Group_gt")

            try:
                x,y = unary_union(RG_Group_gt).exterior.xy
                plt.plot(RG_Group_gt.centroid.x,RG_Group_gt.centroid.y)
                plt.plot(x,y)
            except Exception as e:
                print(e, "no RG GT annotation")
            try:
                cords = T_Group_gt.coords if len(T_Group_gt) > 1 else [0,0]
                plt.plot(cords)
            except Exception as e:
                print(e,"probably no thread GT annotation")
    


        try:
            x,y = unary_union(LG_dl).exterior.xy
            plt.plot(x,y)                
        except Exception as e:
            print(e,"No LG DL label")
        try:
            x,y = unary_union(RG_dl).exterior.xy
            plt.plot(x,y)
        except Exception as e:
            print(e,"No RG DL label") 
        if not isinstance(needleShape,list):
            for ns in needleShape.geoms:
                try:
                    x,y = unary_union(ns).exterior.xy
                    plt.plot(ns.centroid.x,ns.centroid.y)
                    plt.plot(x,y)
                except Exception as e:
                    print(e,"No Needle DL label")
        
        #unaryThread = unary_union(T_dl.geoms)
        #print("thread geoms:",T_dl.geoms,"unaryThread",unaryThread.is_valid,type(unaryThread))

        for thread in T_dl.geoms:
            if thread.exterior.xy:
                #x,y = thread.exterior.xy
                #plt.plot(x,y)
                pass

        strArr = [CtxI.getContext(frameNumber),ctxPredLine,"LG->T:"+"{:.2f}".format(LG_inter_T),"RG->T:"+"{:.2f}".format(RG_inter_T)]
        offset = 1
        for s in strArr:
            x = 10
            y = 22 * offset; 
            plt.text(x,y,s,fontsize=10,color='red')
            offset+=1

        for s in messages:
            x = 10
            y = 22 * offset; 
            plt.text(x,y,s,fontsize=10,color='red')
            offset+=1

        plt.savefig(outputFName)
        plt.close()
                
    def DrawSingleImageContextKT(self, pred, gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T,GT=False):
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt  

        image = cv.imread(imageFName)
        plt.imshow(image, cmap='gray') 
        if GT:
        
            try:
                x,y = unary_union(LG_Group_gt).exterior.xy
                plt.plot(x,y)
            except Exception as e:
                print(e,"No LG_Group_gt")

            try:
                x,y = unary_union(RG_Group_gt).exterior.xy
                plt.plot(x,y)
            except Exception as e:
                print(e, "no RG GT annotation")
            try:
                cords = T_Group_gt.coords if len(T_Group_gt) > 1 else [0,0]
                plt.plot(cords)
            except Exception as e:
                print(e,"probably no thread GT annotation")
    
        try:
            x,y = unary_union(LG_dl).exterior.xy
            plt.plot(x,y)                
        except Exception as e:
            print(e,"No LG DL label")
        try:
            x,y = unary_union(RG_dl).exterior.xy
            plt.plot(x,y)
        except Exception as e:
            print(e,"No RG DL label") 
        #unaryThread = unary_union(T_dl.geoms)
        #print("thread geoms:",T_dl.geoms,"unaryThread",unaryThread.is_valid,type(unaryThread))
        for thread in T_dl.geoms:
            if thread.exterior.xy:
                #print(len(t))
                x,y = thread.exterior.xy
                #print(thread.exterior.xy)
                plt.plot(x,y)

        strArr = [CtxI.getContext(frameNumber),ctxPredLine,"LG->T:"+str(LG_inter_T),"RG->T:"+str(RG_inter_T)]
        offset = 1
        for s in strArr:
            x = 10
            y = 22 * offset; 
            plt.text(x,y,s,fontsize=12,color='red')
            offset+=1
        
        
        
        #plt.show()
        plt.savefig(outputFName)
        plt.close()
        
        
        
        '''
        LG_dl_s = gpd.GeoSeries(LG_dl)
        LG_dl_s.plot()
        RG_dl_s = gpd.GeoSeries(RG_dl)
        RG_dl_s.plot()
        T_dl_s = gpd.GeoSeries(T_dl)
        T_dl_s.plot()

        LG_Group_gt_s = gpd.GeoSeries(LG_Group_gt)
        LG_Group_gt_s.plot()
        RG_Group_gt_s = gpd.GeoSeries(RG_Group_gt)
        RG_Group_gt_s.plot()
        T_Group_gt_s = gpd.GeoSeries(T_Group_gt)
        T_Group_gt_s.plot()
        plt.show()
        '''
 
    def GetKTShapes(self,gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints):
        RG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyRG[i],gtPolyRG[i+1]) for i in range(0,len(gtPolyRG),2)] ) for gtPolyRG in gtPolygons["Right Grasper"]]) if "Right Grasper" in gtPolygons.keys() else []
        LG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyLG[i],gtPolyLG[i+1]) for i in range(0,len(gtPolyLG),2)] ) for gtPolyLG in gtPolygons["Left Grasper"]]) if "Left Grasper" in gtPolygons.keys() else []
        T_Group_gt =  geo.LineString( [point for point in SingleThreadPoints] ) if len(SingleThreadPoints) >1 else []
        #NG_Group = geo.MultiPolygon([ geo.Polygon( [ (gtPolyN[i],gtPolyN[i+1]) for i in range(0,len(gtPolyN),2)] ) for gtPolyN in gtPolygons["Needle"]])
             
        LG_dl = geo.MultiPolygon([ geo.Polygon(poly) for poly in LgrasperPoints])
        RG_dl =  geo.MultiPolygon([ geo.Polygon(poly) for poly in RgrasperPoints])        
        threadPolys = []
        
        for k in range(len(ThreadContours)):     
            if(len(ThreadContours[k])>2):           
                cnt = ThreadContours[k]
                threadPolys.append( geo.Polygon( [(c[0][0],c[0][1]) for c in cnt]))
        print("threadPolys",len(threadPolys))  
        T_dl =   geo.MultiPolygon( threadPolys )
        #RG_dl =  geo.Polygon( [ (LgrasperPoints[i],gtPolyRG[i+1]) for i in range(0,len(gtPolyRG),2)] ) 
        #LG_dl = geo.MultiPolygon([ geo.Polygon( [ (gtPolyLG[i],gtPolyLG[i+1]) for i in range(0,len(gtPolyLG),2)] ) for gtPolyLG in gtPolygons["left Grasper"]])
        #gtRGrasper = geo.Polygon()
        pred = [LG_dl,RG_dl,T_dl]
        gt = [LG_Group_gt,RG_Group_gt,T_Group_gt]
        return pred, gt

    def GetGenShapes(self,gtPolygons,gtKeypoints,SingleThreadPoints,polylineSeries,ThreadContours,LgrasperPoints,RgrasperPoints):
        RG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Right Grasper"]]) if "Right Grasper" in gtPolygons.keys() else []
        LG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Left Grasper"]]) if "Left Grasper" in gtPolygons.keys() else []
        #T_LineGroup_gt = geo.LineString( [point for point in SingleThreadPoints] ) if len(SingleThreadPoints) >1 else []         
               
        polys = []
        for i in range(len(polylineSeries)):
            l = len(polylineSeries)
            SingleThreadX = []
            SingleThreadY = []
            points = []
            for j in range(0,len(polylineSeries[i]),2):
                SingleThreadX.append(polylineSeries[i][j])
                SingleThreadY.append(polylineSeries[i][j+1])
                points.append((polylineSeries[i][j],polylineSeries[i][j+1]))
            if (len(points)>=3):
                currPoly = geo.Polygon(geo.LineString( [point for point in points] ) )
                polys.append(currPoly)

        #SingleThreadPoints = [(SingleThreadX[i],SingleThreadY[i]) for i in range(len(SingleThreadX))]

        T_Group_gt =  geo.MultiPolygon(polys)  

        for poly in LgrasperPoints:
            if(len(poly)<3):
                LgrasperPoints.remove(poly)

        LG_dl = geo.MultiPolygon([ geo.Polygon(poly) for poly in LgrasperPoints])
        for poly in RgrasperPoints:
            if(len(poly)<3):
                RgrasperPoints.remove(poly)
        RG_dl =  geo.MultiPolygon([ geo.Polygon(poly) for poly in RgrasperPoints])
            
        threadPolys = []        
        for k in range(len(ThreadContours)):     
            if(len(ThreadContours[k])>2):           
                cnt = ThreadContours[k]
                threadPolys.append( geo.Polygon( [(c[0][0],c[0][1]) for c in cnt]))
        T_dl =   geo.MultiPolygon(threadPolys)

        pred = [LG_dl,RG_dl,T_dl]
        gt = [LG_Group_gt,RG_Group_gt,T_Group_gt]
        return pred, gt

    def GetRingShapes(self,Ring4Points,Ring5Points,Ring6Points,Ring7Points,gtPolygons):
        #print(Ring4Points)
        R4_Group = geo.MultiPolygon([ geo.Polygon( ring ) for ring in Ring4Points if len(ring)> 3]) if  len(Ring4Points)>0 else []
        R5_Group = geo.MultiPolygon([ geo.Polygon( ring ) for ring in Ring5Points if len(ring)> 3]) if  len(Ring5Points)>0 else []
        R6_Group = geo.MultiPolygon([ geo.Polygon( ring ) for ring in Ring6Points if len(ring)> 3]) if  len(Ring6Points)>0 else []
        R7_Group = geo.MultiPolygon([ geo.Polygon( ring ) for ring in Ring7Points if len(ring)> 3]) if  len(Ring7Points)>0 else []

        R4_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Ring_4"]]) if "Ring_4" in gtPolygons.keys() else []
        R5_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Ring_5"]]) if "Ring_5" in gtPolygons.keys() else []
        R6_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Ring_6"]]) if "Ring_6" in gtPolygons.keys() else []
        R7_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Ring_7"]]) if "Ring_7" in gtPolygons.keys() else []
        
        return [R4_Group,R5_Group,R6_Group,R7_Group],[R4_Group_gt,R5_Group_gt,R6_Group_gt,R7_Group_gt]

    def GetNeedleShapes(self,NeedlePoints,gtPolygons):
        
        N_dl =  geo.MultiPolygon([ geo.Polygon(poly) for poly in NeedlePoints if len(poly)> 3 ]) if len(NeedlePoints)>0 else []
        #print(len(NeedlePoints))
        Ring_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (poly[i],poly[i+1]) for i in range(0,len(poly),2)] ) for poly in gtPolygons["Needle Mask"]]) if "Needle Mask" in gtPolygons.keys() else []
        
        return N_dl,Ring_Group_gt

main()