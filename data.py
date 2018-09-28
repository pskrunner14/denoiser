import os

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

ATTRS_FILE = "datasets/lfw_attributes.txt"  # http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
IMAGES_DIR = "datasets/lfw-deepfunneled"  # http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
RAW_IMAGES_DIR = "datasets/lfw"  # http://vis-www.cs.umass.edu/lfw/lfw.tgz

"""Load Dataset

Loads the Labeled Faces in the Wild dataset
(train/test split) and attributes into memory.
"""
def load_dataset(use_raw=False, dx=80, dy=80, dimx=45, dimy=45):

    # read attrs
    df_attrs = pd.read_csv(ATTRS_FILE, sep='\t', skiprows=1)
    df_attrs.columns = list(df_attrs.columns)[1: ] + ["NaN"]
    df_attrs = df_attrs.drop("NaN", axis=1)
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # read photos
    X = []
    photo_ids = []

    image_dir = RAW_IMAGES_DIR if use_raw else IMAGES_DIR
    folders = os.listdir(image_dir)
    for folder in tqdm(folders, total=len(folders), desc='Preprocessing', leave=False):
        files = os.listdir(os.path.join(image_dir, folder))
        for file in files:
            if not os.path.isfile(os.path.join(image_dir, folder, file)) or not file.endswith(".jpg"):
                continue
            
            # preprocess image
            img = cv2.imread(os.path.join(image_dir, folder, file))
            img = img[dy:-dy, dx:-dx]
            img = cv2.resize(img, (dimx, dimy))
            
            # parse person
            fname = os.path.split(file)[-1]
            fname_splitted = fname[:-4].replace('_', ' ').split()
            person_id = ' '.join(fname_splitted[:-1])
            photo_number = int(fname_splitted[-1])
            if (person_id, photo_number) in imgs_with_attrs:
                X.append(img)
                photo_ids.append({'person': person_id, 'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    X = np.stack(X).astype('uint8')
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)

    IMG_SHAPE = X.shape[1:]

    # center images
    X = (X.astype('float32') / 255.0) - 0.5

    # split
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

    return X_train, X_test, IMG_SHAPE, all_attrs