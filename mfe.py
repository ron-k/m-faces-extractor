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



def classify(path):
    res=DeepFace.analyze(path,actions=("age","gender"), enforce_detection = False)
    return {"age" : res[0]["age"], "gender":res[0]["dominant_gender"]}


def printImg(path):
    if (True):
        return
    #img = cv2.imread(path, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cl=classify(path)
    if (cl["gender"]=='Man'):
        faceLabel="M"
    else:
        faceLabel="F"
    clStr=f"{faceLabel}_{cl['age']}"
    print(f"classify={clStr}")
    #plt.imshow(img)
    #plt.show()

def checkMatch(subj,db):
    results=DeepFace.find(subj, db, enforce_detection = False, silent = True)
    #print(f"DeepFace.find: results={results}")
    #for result in results:
        #for identity in result.identity:
            #print(f"result: {identity}")
            #printImg(identity)
    if (results[0]['VGG-Face_cosine'].empty):
        return 0
    return 1-results[0]['VGG-Face_cosine'][0]


def findMatch(subj,databases):
    printImg(subj)
    found = None
    goldenScore = 0
    silverScore = 0
    scores={"label":[], "score":[]}
    for db in databases:
        label = db['label']
        path = db['path']
        score = checkMatch(subj,path)
        print(f"findMatch: label {label} score={score}")
        if (score >0.45):
            print(f"findMatch: this might be {label} score={score}")
            scores['label'].append(label)
            scores['score'].append(score)
#   print(f"findMatch: scores={scores} scores['label']={scores['label']} scores['score']={scores['score']}")
    if (len(scores['label'])==0):
        found = None
    elif (len(scores['label'])==1):
        found = scores['label'][0]
    else: #several significant scores
        maxScore = max(scores['score'])
        maxScoreIdx = scores['score'].index(maxScore)
        maxLabel = scores['label'][maxScoreIdx]
        
        del scores['label'][maxScoreIdx]
        del scores['score'][maxScoreIdx]
        
        maxScore2 = max(scores['score'])
        
        print(f"findMatch: max={maxLabel} score={maxScore}  nextScore={maxScore2}")
        if (maxScore - maxScore2 > 0.1):
            return maxLabel
        else:
            return None

#    if (found == None):
#        print(f"findMatch: this is unknown")
#    else:
#        print(f"findMatch: this is {found}")
    return found

def createDatabases(dbRoot):
    if (not os.path.exists(dbRoot)):
        raise Exception(f"no such path '{dbRoot}'")
    databases=[]
    for dirName in next(os.walk(dbRoot))[1]:
        path=os.path.join(dbRoot,dirName)
        pickleFile=os.path.join(path,"representations_vgg_face.pkl")
        if os.path.exists(pickleFile):
            os.remove(pickleFile)
        db={"path":path, "label":dirName}
        databases.append(db)
    if (len(databases)==0):
        print(f"no entries in the database")
    return databases

def isFaceKnownLocally(face, db):
    for index, known_face in enumerate(db):
        result = DeepFace.verify(img1_path = face, img2_path = known_face,
                           enforce_detection = False)
#                         print(f"Compared with known face: index={index} result={result}")
        if (result["verified"]):
            return True
    return False

def extractFaces(videoPath,skip,db):
    sTime = time.time()

    videoFileNameNoExt = Path(videoPath).stem
    videoParentDir = PurePath(videoPath).parent
    targetDir = os.path.join(videoParentDir,f"{videoFileNameNoExt}-faces")

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    faces = []
    cap = cv2.VideoCapture(videoPath) # read video file
    if (not cap.isOpened()):
        raise Exception(f"could not open video file '{videoPath}'")
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps)
    path = os.path.basename(videoPath)
    frameIndex = 0

    prevFrameFaces=[]
    while cap.isOpened():
            frameIndex = frameIndex + 1

            ret, frame = cap.read()

            if frameIndex > 10000000:
                break
            if not ret:
                break
            if skip > 0 and frameIndex % skip!=0:
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
    #                     if (isFaceKnownLocally(faceFrame,prevFrameFaces)):
    #                         print(f"Was already processed in previous frame")
    #                         isFaceNew = False
    #                         break
                        if (isFaceKnownLocally(faceFrame,faces)):
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
                    
                    dbMatch=findMatch(faceFrame,db)
                    faceLabel=f"{frameIndex}_{len(faces)}"
                    if (dbMatch== None):
                        faceTargetDir=targetDir
                    else:
                        faceTargetDir= os.path.join(targetDir, dbMatch)
                    print(f"New face found: frame {frameIndex}: confidence={confidence} label={faceLabel}")
                    #plt.imshow(faceFrame)
                    #plt.show()
                    writeFace(faceFrame,faceLabel,faceTargetDir,videoFileNameNoExt)
                    faces.append(faceFrame)
                    
                
            prevFrameFaces = thisFrameFaces
            if (frameIndex % 10 == 0) and frameIndex > 1:
                elapsed =  time.time()  - sTime
                fps = frameIndex / elapsed
                print(f"Total frames processed {frameIndex} ({frameIndex*100/ totalFrames}%) faces={len(faces)} elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))} fps={fps}")
                

def writeFace(face, faceLabel ,targetDir, origin):
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    targetFilename=f"{origin}_{faceLabel}.png"
    outputPath = os.path.join(targetDir, targetFilename)
    cv2.imwrite(outputPath, face[:, :, ::-1] * 255)

def main(args):

    videoPath = args["input"]
    skip = int(args["skip"])
    dbPaths = args["database"]
    db=createDatabases(dbPaths)
    extractFaces(videoPath,skip,db)
 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-db", "--database",  required=True, help="Path to database")
  parser.add_argument("-s", "--skip", default=0, help="amount of frames to skip in between captures")
  args = vars(parser.parse_args())
  main(args)