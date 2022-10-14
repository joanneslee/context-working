import cv2 as cv
import numpy as np
import glob


import os
import utils
import pathlib

colors =["#E36D6D","#5E81B5","#D47BC9","#7CEB8E","#C9602A","#77B9E0","#A278F0","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5","#D47BC9","#7CEB8E","#E36D6D","#C9602A","#77B9E0","#A278F0","#D66F6D","#5E81B5","#D47BC9","#FAB6F4","#C9602A","#E09C24","#EA5536","#A1C738","#5E81B5"]

def main():
    task = "Suturing"
    I = VideoInterface(task)
    I.makevideos()

class VideoInterface:

    def __init__(self, task):
        self.CWD = os.path.dirname(os.path.realpath(__file__))        
        self.task = task
        self.imagesDir = os.path.join(self.CWD, task,"images")    
        self.deeplabOutputDir= os.path.join(self.CWD,task,"deeplab_labeled_images")
        self.OS = "windows" 

    def makevideos(self ):
        Dirs = []
        for root, dirs, files in os.walk(self.imagesDir):
            Dirs = dirs
            break
        print("Trials:",Dirs)
        TrialNum = 0
        for Trial in Dirs:
            TrialRoot = os.path.join(self.CWD,self.task,"deeplab_labeled_images",Trial)
           
            img_array = []
            for filename in glob.glob(TrialRoot+'/*.png'):
                img = cv.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

            out = cv.VideoWriter(self.CWD+'/project'+Trial+'_slow.avi',cv.VideoWriter_fourcc(*'DIVX'), 15, size)
            
            for i in range(len(img_array)):
                for j in range(3):
                    out.write(img_array[i])
            
            out.release()
            print("Saved Video",Trial)
            TrialNum+=1

        print("Processed ",TrialNum,"trials")

main();
