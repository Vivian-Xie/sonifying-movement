import cv2
import time
import mediapipe as mp
import sys

# import library ---------------------------------------------------------------
import pygame.midi
import threading
from queue import Queue
import copy

import numpy as np
import math

class poseDetector():

    def __init__(self, static_image_mode = False,
                 model_complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation = False,
                 smooth_segmentation = True,
                 min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    if cx<480:
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                        cv2.putText(img, str(lmList[-1][0]), (cx,cy+10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    else:
                        cv2.circle(img, (cx, cy), 3, (255, 255, 0), cv2.FILLED)
                        cv2.putText(img, str(lmList[-1][0]), (cx,cy+10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        return lmList



"""
# define all the constant values -----------------------------------------------
device = 0     # device number in win10 laptop
instrument = 9 # http://www.ccarh.org/courses/253/handout/gminstruments/
note_Do = 48   # http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm
note_Re = 50
note_Me = 52
volume = 127
wait_time = 0.5

# initize Pygame MIDI ----------------------------------------------------------
pygame.midi.init()

# set the output device --------------------------------------------------------
player = pygame.midi.Output(device)

# set the instrument -----------------------------------------------------------
player.set_instrument(instrument)

player.note_on(note_Re, volume)
time.sleep(wait_time)
player.note_off(note_Re, volume)

player.note_on(note_Me, volume)
time.sleep(wait_time)
player.note_off(note_Me, volume)
"""



 
def pose(q):
    #cap = cv2.VideoCapture("../PoseVideos/2.mp4")
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    cycle=0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        cycle+=1
        if cycle==24:
            q.put(lmList)
            cycle=0

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            #exit()
            sys.exit("program stop")
            break

def play_midi(q):
    
    device = 0     # device number in win10 laptop
    instrument = 9 # http://www.ccarh.org/courses/253/handout/gminstruments/
    note_Do = 48   # http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.html
    note_Re = 50
    note_Me = 52
    volume = 127 
    wait_time = 0.5
    
    while True:
        lmList=q.get()
        pygame.midi.init()
        # set the output device --------------------------------------------------------
        player = pygame.midi.Output(device)
        # set the instrument -----------------------------------------------------------
        player.set_instrument(instrument)
        print(lmList)
        if lmList[20]:
            """
            if lmList[20][1]<480:
                player.note_on(note_Re, volume)
                time.sleep(wait_time)
                player.note_off(note_Re, volume)
            else:
                player.note_on(note_Do, volume)
                time.sleep(wait_time)
                player.note_off(note_Do, volume)
            """
            note=pos_2_note(lmList[20][1],0,127)
            player.note_on(note, volume)
            time.sleep(wait_time)
            player.note_off(note, volume)
            time.sleep(0.03)
        pygame.midi.quit()
        del player

def pos_2_note(data,MIN,MAX):
    return int(MIN +(MAX-MIN)/960*data)




def main():
    q=Queue()
    
    thread1=threading.Thread(target=pose,name="T1",args=(q,))
    thread2=threading.Thread(target=play_midi,name="T2",args=(q,))
    thread1.start()
    thread2.start()



if __name__ == "__main__":
    main()
    
# close the device -------------------------------------------------------------
   