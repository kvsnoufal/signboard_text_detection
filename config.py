import pickle as cp
import pandas as pd
import glob
# CONFIG
FILTERED_IMG_NAME_PATH = './input/imnames.cp.1'
with open(FILTERED_IMG_NAME_PATH, 'rb') as f:
        filtered_imnames = set(cp.load(f))
BGS =  list(filtered_imnames)
BGS = ["./input/bg_img/bg_img/"+t for t in BGS] 
DFTEXT = pd.read_csv("./input/text/english_arabic.csv")
ENGLISH_WORDS = DFTEXT["in English"].tolist()
ARABIC_WORDS = DFTEXT["Arabic"].tolist()
PROB_UPPERCASE = 0.5
ENGLISH_FONTS = glob.glob("./input/font_cp/*.ttf")
ARABIC_FONTS = glob.glob("./input/font_ar/*.ttf")
TEXTURES = glob.glob("./input/textures/archive/dtd/images/*/*.jpg")
MAX_NUMBER_OF_WORDS = 4
OPTIONS = ["ENGLISH","ARABIC","ENGLISH AND ARABIC"]
NUM_IMAGES = 100
OUTPUTFOLDER = "imagefolder"

##################