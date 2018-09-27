#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import random
import string
import numpy as np

def makeimg(tmb=0, fake_late=20):
    fcol = [0,0,0]
    tcol = [255,255,255]

#    line = [gcol] * 12
#    img = [line] * 12
    img=[]
    for i in range(0,3):
        if i == tmb:
            tmp = tcol
            tcol = fcol
            fcol = tmp
        for j in range(0,4):
            line=[]
            for k in range(0,12):
                gorw = random.randrange(fake_late)
                col = tcol
                if gorw == 0:
                    col = fcol

                line.append(col)
            img.append(line)
        if i == tmb:
            tmp = tcol
            tcol = fcol
            fcol = tmp
    return(img)

def gen_rand_str(length, chars=None):
    if chars is None:
        chars = string.digits + string.ascii_letters
    return ''.join([random.choice(chars) for i in range(length)])



amount = 10

for bmt in range(0,3):
    fl = 20
    if bmt == 0:
        dirname = "/export/space/araki-t/Make3class/top/"
    elif bmt == 1:
        dirname = "/export/space/araki-t/Make3class/middle/"
    elif bmt == 2:
        dirname = "/export/space/araki-t/Make3class/bottom/"
#    else : #no use (only testcase)
#    fl = 10
#    dirname="/export/space/araki-t/Make3class/test/"

    print("cooking "+str(amount)+" images to "+dirname)
    for i in range(0,amount):
        imgname=(dirname+gen_rand_str(8)+".png")
        img = makeimg(bmt, fl)
        cv2.imwrite(imgname, np.array(img))
