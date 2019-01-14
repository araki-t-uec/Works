import sys, re

receive_file = sys.argv

if len(receive_file) > 1:
    f = open(receive_file[1])
else:
    print("./hoge.py piyo.txt")
    exit()

n = 6 # 30 fps / 5 = 6fps

for aline in f:
    text = aline.split()[1].split("_")[-1].split(".")[0]
    if int(text) %n == 0:
        print(aline.strip())
    
