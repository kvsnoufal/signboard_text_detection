import cv2
import glob
import pandas as pd
import numpy as np
from utils import *
import os
from config import *




if __name__ == "__main__":
    os.makedirs(OUTPUTFOLDER,exist_ok=True)
    os.makedirs(os.path.join(OUTPUTFOLDER,"genimages"),exist_ok=True)
    os.makedirs(os.path.join(OUTPUTFOLDER,"genmasks"),exist_ok=True)
    
    
    for i in tqdm(range(NUM_IMAGES),total=NUM_IMAGES):
        try:
            shape,img_mask,bg,eng_phrase,ar_phrase = step1()
            op = np.random.choice(OPTIONS)
            img, eng_bboxmask, ar_bboxmask  = step2(shape,img_mask,eng_phrase,ar_phrase,op)

            overlaytexture,eng_bboxmaks_adj,ar_bboxmaks_adj,w,h = step3(eng_bboxmask,ar_bboxmask,img_mask,img)
            bg_img,reboxed_rotated_scaled_eng_bboxmaks_adj,reboxed_rotated_scaled_ar_bboxmaks_adj,xb,yb,wb,hb,rotated,rotated_scaled_img_mask = \
                transform(bg,img_mask,overlaytexture,h,w,eng_bboxmaks_adj,ar_bboxmaks_adj,op)
            bkbg_img,eng_mask_bkbg_img,ar_mask_bkbg_img,roi = \
                step4(bg_img,reboxed_rotated_scaled_eng_bboxmaks_adj,reboxed_rotated_scaled_ar_bboxmaks_adj,xb,yb,wb,hb,rotated,rotated_scaled_img_mask,op)
        except:
            print(f"Error at {i}")
            continue

        cv2.imwrite(os.path.join(OUTPUTFOLDER,"genimages",f"{i}.jpg"),bkbg_img)
        if type(eng_mask_bkbg_img) != type(None):
            cv2.imwrite(os.path.join(OUTPUTFOLDER,"genmasks",f"eng_{i}.jpg"),eng_mask_bkbg_img)
        if type(ar_mask_bkbg_img) != type(None):
            cv2.imwrite(os.path.join(OUTPUTFOLDER,"genmasks",f"ar_{i}.jpg"),ar_mask_bkbg_img)
