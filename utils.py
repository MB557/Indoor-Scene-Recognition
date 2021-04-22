# Basic libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle as pkl

labels = ['kitchen', 'livingroom', 'bedroom', 'airport_inside', 'casino', 
          'warehouse', 'bakery', 'bookstore', 'toystore', 'bathroom']

def resize(S, dest):
    '''
        1) Making a list of images
        2) Making a list of labels
        3) Resizing the images to SxSx3
        4) Converting the lists into numpy arrays

        params : 
            S -> Resultant Size after Resizing

        return : 
            numpy_array -> contains the images (n, S, S, 3)
            scenes      -> contains the clases (n,)
    '''
    array_of_imgs = []
    to_be_deleted = []
    Y = []

    path_to_each_image = []
    for dirs in labels:
        for images in sorted(os.listdir(dest + '/' + dirs)):
            path_to_each_image.append('/'+ dirs + '/' + images)

    for sub_path in tqdm(path_to_each_image):
        path = dest + sub_path
        img = cv2.imread(path)

        try :
            # resize to SxSx3
            new_img = cv2.resize(img, (S,S), interpolation = cv2.INTER_AREA)
        except:
            to_be_deleted.append(path)
            continue
        
        # inserting into the array
        array_of_imgs.append(new_img)
        # inserting into labels
        Y.append(path.split('/')[-2])

    # convert lists to numpy arrays
    numpy_array = np.array(array_of_imgs)
    scenes = np.array(Y)

    return numpy_array, scenes

def dump_into_pkl(data, name):
    '''
        dumps the data into a pkl file
        
        params : 
            data -> whatever you want to dump
            name -> name of the file
    '''
    outfile = open(name,'wb')
    pkl.dump(data, outfile)
    outfile.close()
    
    print(name + " dumped")


def load_from_pkl(name):
    '''
        loads pkl data from the .pkl file
        
        params : 
            name -> name of the file

        return :
            X    -> whatever data was inside the pkl file
    '''
    infile = open(name,'rb')
    X = pkl.load(infile)
    infile.close()

    return X

def prep_labels(scenes):
    '''
        prepares scenes to fit as labels into PCA (str -> int)

        params :
            scenes -> the labels in string format
        return : 
            void
    '''
    labels = sorted(list(set(scenes)))
    label_map = {label:i for i,label in enumerate(labels)}

    for i in range(scenes.shape[0]):
        scenes[i] = int(label_map[scenes[i]])

    scenes = scenes.astype(np.int)
    return scenes