import os
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
#import geopandas as gpd

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
    #I.DrawDeepLab() #TODO: get ctx lines like consensus
    I.GenerateContext(SAVE=True)
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
                #if "frame" not in file:
                tissueFname = os.path.join(root,file)
                T = utils.ViaJSONInterface(tissueFname)
                Tissues[file.replace(".json","")] = T.getDataDict()
                TissueFrames[file.replace(".json","")] = T.getFrameNumbers()
                print("Input: Tissue Keypoints",file)

        for root, dirs, files in os.walk(self.grasperJawDir):
            for file in files:
                #if "frame" not in file:
                jawsFname = os.path.join(root,file)      
                J = utils.ViaJSONInterface(jawsFname)
                GrasperJaws[file.replace(".json","")] = J.getDataDict()
                GrasperFrames[file.replace(".json","")] = J.getFrameNumbers()
                print("Input: Grasper Jaws",file)

        for root, dirs, files in os.walk(self.contourDir):
            for file in files:
                #if "frame" not in file:
                label_class = os.path.basename(root)
                VIAFname = os.path.join(root,file)
                J = utils.VIAPolyJSON(VIAFname)
                class_name = J.getClassKey(label_class)
                if file.replace(".json","") not in DeeplabVIAPoints.keys():
                    DeeplabVIAPoints[file.replace(".json","")] = {}
                if file.replace(".json","") not in DeeplabVIAFrames.keys():
                    DeeplabVIAFrames[file.replace(".json","")] = {}
                DeeplabVIAPoints[file.replace(".json","")][class_name] = J.getDataDict()
                DeeplabVIAFrames[file.replace(".json","")][class_name] = J.getFrameNumbers()
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

                    try:
                        NeedlePoints = All_dl_points["dl_grasper_L"][str(frameNumber)]
                    except Exception as e:
                        print(e,"in RgrasperPoints")
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
                    SingleThreadPoints = [(SingleThreadX[i],SingleThreadY[i]) for i in range(len(SingleThreadX))]
                    for i in range(len(polylineSeries)):
                        l = len(polylineSeries)
                        for j in range(0,len(polylineSeries[i]),2):
                            SingleThreadX.append(polylineSeries[i][j])
                            SingleThreadY.append(polylineSeries[i][j+1])

                    TContours = self.getThreadContours(threadMaskImage)
                    ThreadContours = []
                    areas = []
                    largestIndex = -1
                    largestArea = 0
                    minArea = 80
                    for k in range(len(TContours)):                    
                        cnt = TContours[k]
                        area = cv.contourArea(cnt)
                        areas.append(area)
                        if area>largestArea:
                            largestIndex=k
                            largestArea=area
                        if area >= minArea:
                            ThreadContours.append(cnt)
                    print("Areas",areas)

                    try:
                        L_Dist = utils.distTwoPoints(GrasperJawPoints[2],GrasperJawPoints[3])
                        R_Dist = utils.distTwoPoints(GrasperJawPoints[0],GrasperJawPoints[1])
                        Grasper_Dist,Grasper_DistX,Grasper_DistY = self.distGraspers(GrasperJawPoints)
                    except Exception as e:
                        L_Dist = 0
                        R_Dist = 0 
                    PARAM_JAW_DIST = 10
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
                        

                    # maskify Thread
                    if("Knot" in self.task):
                        pred, gt = self.GetKTShapes(gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints)
                        ctxPredLine, LG_inter_T, RG_inter_T = self.GenerateContextLineKT(pred, gt ,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX)
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)

                        #self.DrawSingleImageContextKT(pred, gt,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,L_Gripping,R_Gripping,LG_inter_T, RG_inter_T)
                   
                    '''
                    
                    if(not os.path.isdir(outputRoot)):
                        path = pathlib.Path(outputRoot)
                        path.mkdir(parents=True, exist_ok=True)                
                    if("Knot" in self.task):
                        inter_couts,bool_inter = self.GetKTIntersections(L_grasper,R_grasper,Thread)
                        [LG_inter_T,RG_inter_T] = inter_couts
                        ctxPredLine = self.GenerateContextLineKT(LG_inter_T,RG_inter_T,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX)
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        #self.DrawSingleImageContextKT(L_grasper,R_grasper,Thread,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,inter_couts,Grasper_DistX,L_Gripping,R_Gripping)
                    if("Needle" in self.task):
                        inter_couts,bool_inter,LocalRings = self.GetNPIntersections(L_grasper,R_grasper,Thread,Needle,Rings)       
                        (L_y_center,L_x_center) = ndimage.center_of_mass(L_grasper)
                        (R_y_center,R_x_center) = ndimage.center_of_mass(R_grasper)
                        ROI_y = abs(R_y_center-L_y_center)/2
                        ROI_x = abs(R_x_center-L_x_center)/2
                        needleSum = np.sum(Needle)
                        ctxPredLine = self.GenerateContextLineNP(inter_couts,L_Gripping,R_Gripping,frameNumber,contextLines,Needle,LocalRings,needleSum,ROI_x, ROI_y)
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        # TODO:
                        self.DrawSingleImageContextNP(L_grasper,R_grasper,Thread,Needle,Rings,GrasperJawPoints,imageFName,outputFName,CtxI,ctxPredLine,frameNumber,inter_couts,LocalRings)
                    
                    if("Suturing" in self.task):
                        LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist = self.GetSuturingIntersections(L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred)

                        ctxPredLine = self.GenerateContextLineS(LG_inter_T,RG_inter_T,LG_inter_N,RG_inter_N,N_inter_TS,L_Gripping,R_Gripping,min_Tissue_Dist,needle_center_dist,frameNumber)
                        contextLines.append(ctxPredLine)
                        print(Trial,frameNumber,ctxPredLine)
                        self.DrawSingleImageContextS(L_grasper,R_grasper,Thread,Needle,TissuePoints,GrasperJawPoints,imageFName,outputFName,CtxI,CtxI_Pred,ctxPredLine,frameNumber,L_Dist,R_Dist,min_Tissue_Dist,needle_center_dist)
                    '''
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

    def GenerateContextLineKT(self,pred, gt ,L_Gripping,R_Gripping,frameNumber,contextLines,Grasper_DistX, GT=False):
        [LG_dl,RG_dl,T_dl] = pred
        [LG_Group_gt,RG_Group_gt,T_Group_gt] = gt  
        if GT:
            RG_inter_T = 0
        else:
            RG_inter_T = RG_dl.distance(T_dl)
            LG_inter_T = LG_dl.distance(T_dl)
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
        INTER_THRESH = 2
        
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
                if Grasper_DistX > 100:                    
                    Extra_State = 2 # lose
                else:
                    Extra_State = 1
            elif last == "2":
                if Grasper_DistX > 250:
                    Extra_State = 3 # tight
                else:
                    Extra_State = 2 # lose
            elif last == "3":
                if not L_Gripping or not R_Gripping:
                    Extra_State = 0 # tight
                else:
                    Extra_State = 3 # lose 
        return ""+ str(frameNumber) + " " + str(L_G_Hold) + " " + str(L_G_Touch) + " " + str(R_G_Hold) + " " + str(R_G_Touch) + " " + str(Extra_State), LG_inter_T, RG_inter_T

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


    def getThreadContours(self,threadMaskImage):
        im = cv.imread(threadMaskImage)
        imgray = cv.cvtColor(im,cv.COLOR_RGB2GRAY,0)
        ret, thresh = cv.threshold(imgray, 1, 255, 0)
        ThreadContours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) 
        return ThreadContours

    def GetKTShapes(self,gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints):        
        print(gtPolygons.keys())
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

    def GetSIntersections(self,gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints,NeedlePoints):
        print(gtPolygons.keys())
        RG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyRG[i],gtPolyRG[i+1]) for i in range(0,len(gtPolyRG),2)] ) for gtPolyRG in gtPolygons["Right Grasper"]])
        LG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyLG[i],gtPolyLG[i+1]) for i in range(0,len(gtPolyLG),2)] ) for gtPolyLG in gtPolygons["left Grasper"]])
        NG_Group = geo.MultiPolygon([ geo.Polygon( [ (gtPolyN[i],gtPolyN[i+1]) for i in range(0,len(gtPolyN),2)] ) for gtPolyN in gtPolygons["Needle"]])        
        
        LG_dl = geo.MultiPolygon([ geo.Polygon(poly) for poly in LgrasperPoints])    
        RG_dl =  geo.MultiPolygon([ geo.Polygon(poly) for poly in RgrasperPoints])   
        N_dl =  geo.MultiPolygon([ geo.Polygon(poly) for poly in NeedlePoints])  
        #gtRGrasper = geo.Polygon()
        pass

    def GetNPIntersections(self,gtPolygons,gtKeypoints,SingleThreadPoints,ThreadContours,LgrasperPoints,RgrasperPoints,needlePoints):
        print(gtPolygons.keys())
        RG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyRG[i],gtPolyRG[i+1]) for i in range(0,len(gtPolyRG),2)] ) for gtPolyRG in gtPolygons["Right Grasper"]])
        LG_Group_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPolyLG[i],gtPolyLG[i+1]) for i in range(0,len(gtPolyLG),2)] ) for gtPolyLG in gtPolygons["left Grasper"]])
        NG_Group = geo.MultiPolygon([ geo.Polygon( [ (gtPolyN[i],gtPolyN[i+1]) for i in range(0,len(gtPolyN),2)] ) for gtPolyN in gtPolygons["Needle"]])        
        
        R4_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPoly[i],gtPoly[i+1]) for i in range(0,len(gtPoly),2)] ) for gtPoly in gtPolygons[""]])
        R5_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPoly[i],gtPoly[i+1]) for i in range(0,len(gtPoly),2)] ) for gtPoly in gtPolygons[""]])
        R6_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPoly[i],gtPoly[i+1]) for i in range(0,len(gtPoly),2)] ) for gtPoly in gtPolygons[""]])        
        R7_gt = geo.MultiPolygon([ geo.Polygon( [ (gtPoly[i],gtPoly[i+1]) for i in range(0,len(gtPoly),2)] ) for gtPoly in gtPolygons[""]])        
        
        
        #RG_dl =  geo.Polygon( [ (LgrasperPoints[i],gtPolyRG[i+1]) for i in range(0,len(gtPolyRG),2)] ) 
        #LG_dl = geo.MultiPolygon([ geo.Polygon( [ (gtPolyLG[i],gtPolyLG[i+1]) for i in range(0,len(gtPolyLG),2)] ) for gtPolyLG in gtPolygons["left Grasper"]])
        #gtRGrasper = geo.Polygon()
        pass



main()