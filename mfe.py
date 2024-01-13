#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from pathlib import PurePath
import argparse
import cv2
from matplotlib import pyplot as plt
from deepface import DeepFace
import numpy as np
import math
import time


def classify(img):
    res=DeepFace.analyze(img,actions=("age","gender"), enforce_detection = False, detector_backend='retinaface')
    print(f"img={img} res={res}")
    return {"age" : res[0]["age"], "gender":res[0]["dominant_gender"]}

    
def isFaceKnown(face, db):
    for index, known_face in enumerate(db):
        result = DeepFace.verify(img1_path = face, img2_path = known_face,
                           enforce_detection = False)
#                         print(f"Compared with known face: index={index} result={result}")
        if (result["verified"]):
            return True
    return False

def extract_faces(videoPath,skip):
    sTime = time.time()

    videoFileNameNoExt = Path(videoPath).stem
    videoParentDir = PurePath(videoPath).parent
    targetDir = os.path.join(videoParentDir,f"{videoFileNameNoExt}-faces")

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    faces = []
    cap = cv2.VideoCapture(videoPath) # read video file
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps)
    path = os.path.basename(videoPath)
    i = 0

    prevFrameFaces=[]
    while cap.isOpened():
            i = i + 1

            ret, frame = cap.read()

            if i > 10000000:
                break
            if not ret:
                break
            if skip > 0 and i % skip!=0:
                continue

            face_props = DeepFace.extract_faces(img_path = frame,
                                             #  target_size = (524, 524),
                                             #  detector_backend = "retinaface",
                                               enforce_detection = False,
                                               align = False
                                               )
            thisFrameFaces=[]
            for face_prop in face_props:
                faceFrame  = face_prop['face']
                #faceFrame  = cv2.cvtColor(face_prop['face'], cv2.COLOR_BGR2RGB)
                #faceFrame  = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2RGB)
                confidence = face_prop['confidence']
                if confidence > 0.998:
    #                 print(f"Face found: frame {i}: confidence={confidence}")
    #                 plt.imshow(faceFrame)
    #                 plt.show()
                    if (len(faces)>0):
                        isFaceNew = True
    #                     if (isFaceKnown(faceFrame,prevFrameFaces)):
    #                         print(f"Was already processed in previous frame")
    #                         isFaceNew = False
    #                         break
                        if (isFaceKnown(faceFrame,faces)):
    #                         print(f"Is already known")
                            isFaceNew = False
                            break
                    else:
                        isFaceNew = True
                    thisFrameFaces.append(faceFrame)
                    if (not isFaceNew):
                        continue
                    # begin: disabled until fixed
                    #cl=classify(faceFrame)
                        
                    #if (cl["gender"]=='Man'):
                    #    gender="M"      
                    #else:
                    #    gender="F"
                    #faceLabel=f"{gender}{len(faces)}_{cl['age']}yo"
                    # end: disabled until fixed
                    faceLabel=f"face_{len(faces)}"
                    print(f"New face found: frame {i}: confidence={confidence} label={faceLabel}")
                    #plt.imshow(faceFrame)
                    #plt.show()
                    writeFace(faceFrame,faceLabel,targetDir,videoFileNameNoExt)
                    faces.append(faceFrame)
                    
                
            prevFrameFaces = thisFrameFaces
            if (i % 10 == 0) and i > 1:
                elapsed =  time.time()  - sTime
                fps = i / elapsed
                print(f"Total frames processed {i} faces={len(faces)} elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))} fps={fps}")
                

def writeFace(face, faceLabel ,targetDir, origin):

        targetFilename=f"{faceLabel}_{origin}.png"
        outputPath = os.path.join(targetDir, targetFilename)
        cv2.imwrite(outputPath, face[:, :, ::-1] * 255)

def main(args):

    video_path = args["input"]
    skip = int(args["skip"])
    extract_faces(video_path,skip)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-s", "--skip", default=0, help="amount of frames to skip in between captures")
  args = vars(parser.parse_args())
  main(args)