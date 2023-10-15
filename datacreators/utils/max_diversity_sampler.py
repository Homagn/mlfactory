# Goes through multiple folder containing possibly extermely visually similar images
# Condenses all such images into as small number of images as possible while trying to keep maximum visual diversity
# result folder can be used directly as a pytorch dataloader using imagefolder class
# To implement - some kind of smart clustering algorithm

import cv2
import glob, os
import shutil

def rate_sampler(sdirs, tdir, rate = 10): #simple sample every k frames for high fps game recordings
    if(os.path.isdir(tdir)):
        print("Target folder already exists, deleting ")
        shutil.rmtree(tdir)
        os.mkdir(tdir)
    else:
        print("Target directory not found creating ")
        os.mkdir(tdir)


    for s in range(len(sdirs)):
        sdir = sdirs[s]
        print("changing to ",sdirs[s])
        os.chdir(sdir)

        fidx = []
        for file in sorted(glob.glob("*.png")):
            f = file[:file.find(".png")]
            fidx.append(int(f))

        fidx.sort()
        #rate = 10
        folder_name = "game"+str(s)
        os.mkdir(tdir+"/"+folder_name)
        for i in range(0, len(fidx), rate):
            im = cv2.imread(sdir+"/"+str(i)+".png")
            cv2.imwrite(tdir+"/"+folder_name+"/"+str(s)+"_"+str(i)+".png",im)

        print("sampled ",len(fidx)//rate, " number of files ")




if __name__ == '__main__':
    tdir = "/datasets/behavior_cloning/maze_game/sampled"

    game_idxs = [1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15]
    sdirs = ["/datasets/behavior_cloning/maze_game/game"+str(g) for g in game_idxs]

    rate_sampler(sdirs, tdir)
    
    


