import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import create_dir

""" Global parameters """
image_h = 64 # 512
image_w = 64 # 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # 1
    # create_dir("test/masks")
    # 2
    create_dir("test/test2/masks")

    """ Loading model """
    model = tf.keras.models.load_model("files/model.keras")
    # model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    # 1
    # data_x = glob(os.path.join('test', 'images', '*')) #"test/images/*")
    # 2
    data_x = glob(os.path.join('test', 'test2', 'images', '*')) #"test/images/*")

    for path in tqdm(data_x, total=len(data_x)):

        """ Extracting name """
        # 1
        # name = path.split("/")[-1].split(".")[0]
        
        # 2
        output_imgname = path.split('\\')[-1]
        output_imgpath = os.path.join('test', 'test2', 'masks', output_imgname)

        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (image_w, image_h))
        x = x/255.0
        x = x.astype(np.float32) ## (h, w, 3)
        x = np.expand_dims(x, axis=0) ## (1, h, w, 3)

        """ Prediction """
        y = model.predict(x, verbose=0)[0][:,:,-1]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)

        """ Save the image """
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cat_images = np.concatenate([image, line, masked_image], axis=1)

        # cv2.imwrite(f"test/masks/{name}.png", cat_images)
        # 1
        # cv2.imwrite(f"{name}.png", cat_images)

        # 2
        cv2.imwrite(f"{output_imgpath}", cat_images)