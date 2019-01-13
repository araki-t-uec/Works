import numpy as np
import cv2
import os, sys


savedir="../data/Dogs/Resque/Labeled/Cv2_Optical/"

filelist = []
#receive_dir = sys.argv
# if len(receive_dir) > 1:
#     video_dir = receive_dir[1]
#     paths = os.listdir(video_dir)
#     for j in paths:
#         filelist.append(os.path.join(video_dir, j))
# else:
#     video_dir="../data/Dogs/Resque/Annotated"
#     paths = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
#     for i in paths:
#         for j in os.listdir(os.path.join(video_dir, i)):
#             filelist.append(os.path.join(video_dir, i, j))
filelist = ["../data/Dogs/Resque/20150801.mp4", "../data/Dogs/Resque/20160710.mp4", "../data/Dogs/Resque/20161111.mp4", "../data/Dogs/Resque/2016112701.mp4", "../data/Dogs/Resque/2016112702.mp4", "../data/Dogs/Resque/2017061901.mp4", "../data/Dogs/Resque/2017061902.mp4"]
print(filelist)

## filelist ---> [ '../data/eat-drink/361.mp4', '../data/bark/ts340.mp4',...]
for filename in filelist:
    fatherpath =  os.path.join(savedir, filename.split("/")[-1].split(".")[0])
    cmd = 'mkdir -p ' + fatherpath
    os.system(cmd)
    print(filename, " to ", fatherpath)
    cap = cv2.VideoCapture(filename)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(old_frame)
    hsv[...,1] = 255
    
#    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    counter = 0
    ret,frame = cap.read() # 2nd frame
    while(ret):
        counter += 1
    
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(old_gray ,frame_gray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            
            #p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        except:
            print("Error : cv2.calcOpricalFlowPyrLK: skip this file")
            ercmd = 'echo "ERROR FILE: '+filename+'" >> cantOpticalMp4list'
            os.system(ercmd)
            break
        #if (int(st[0][0])) == 0:
        #    break
        # Select good points
        #good_new = p1[st==1]
        #good_old = p0[st==1]
        # draw the tracks
        #for i,(new,old) in enumerate(zip(good_new,good_old)):
        #    a,b = new.ravel()
        #    c,d = old.ravel()
        #    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        #    img = cv2.add(frame,mask)
           
        #cv2.imshow('frame',img)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break

    
#        savepath = os.path.join(fatherpath, filename.split("/")[-1].split(".")[0])
        savepath = fatherpath
        savename = filename.split("/")[-1].split(".")[0]+"_%06d.jpg"%counter
        savefile = os.path.join(savepath, savename)
        print(savefile)
        cmd = 'mkdir -p ' + savepath
        os.system(cmd)
        
        #cv2.imwrite(savefile, img)
        cv2.imwrite(savefile, rgb)
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        #p0 = good_new.reshape(-1,1,2)

        # Next frame
        ret,frame = cap.read()

#    cv2.destroyAllWindows()
    cap.release()
