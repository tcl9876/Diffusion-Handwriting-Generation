import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle
import argparse
import string
from utils import Tokenizer

'''
Creates the online and offline dataset for training

Before running this script, download the following things from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset

extract these contents and put them in the ./data directory (unless otherwise specified)
they should have the same names, e.g. "lineStrokes-all" (unless otherwise specified)
'''

def remove_whitespace(img, thresh, remove_middle=False):
    #removes any column or row without a pixel less than specified threshold
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)
    
    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle: return img[rows[0]][:, cols[0]]
    else: 
        rows, cols = rows[0], cols[0]
        return img[rows[0]:rows[-1], cols[0]:cols[-1]]
        
def norms(x): 
    return np.linalg.norm(x, axis=-1)

def combine_strokes(x, n):
    #consecutive stroke vectors who point in similar directions are summed
    #if the pen was picked up in either of the strokes,
    #we pick up the pen for the combined stroke
    s, s_neighbors = x[::2, :2], x[1::2, :2]
    if len(x)%2 != 0: s = s[:-1]
    values = norms(s) + norms(s_neighbors) - norms(s + s_neighbors)
    ind = np.argsort(values)[:n]
    x[ind*2] += x[ind*2+1]
    x[ind*2, 2] = np.greater(x[ind*2, 2], 0)
    x = np.delete(x, ind*2+1, axis=0)
    x[:, :2] /= np.std(x[:, :2])
    return x
 
def parse_page_text(dir_path, id):
    dict = {}
    f = open(dir_path + '/' + id)
    has_started = False
    line_num = -1
    for l in f.readlines():
        if 'CSR' in l: has_started = True
        #the text under 'CSR' is correct, the one labeled under 'OCR' is not
        if has_started:
            if line_num>0: # theres one space after 'CSR'
                dict[id[:-4]+ '-%02d' % line_num]  = l[:-1]
                # add the id of the line -0n as a key to dictionary, 
                # with value of the line number (excluding the last \n)
            line_num += 1
    return dict

def create_dict(path):
    #creates a dictionary of all the line IDs and their respective texts
    dict = {}
    for dir in os.listdir(path):
        dirpath = path + '/' + dir
        for subdir in os.listdir(dirpath):
            subdirpath = dirpath + '/' + subdir
            forms = os.listdir(subdirpath)
            [dict.update(parse_page_text(subdirpath, f)) for f in forms]
    return dict
 
def parse_stroke_xml(path):
    xml = open(path)
    xml = xml.readlines()
    strokes = []
    previous = None
    for i, l in enumerate(xml):
        if 'Point' in l:
            x_ind, y_ind, y_end = l.index('x='), l.index('y='), l.index('time=')
            x = int(l[x_ind+3:y_ind-2])
            y = int(l[y_ind+3:y_end-2])
            is_end = 1.0 if '/Stroke' in xml[i+1] else 0.0
            if previous is None: previous = [x, -y]
            else:
                strokes.append([x - previous[0], -y - previous[1], is_end])
                previous = [x, -y]
    
    strokes = np.array(strokes)
    strokes[:, 2] = np.roll(strokes[:, 2], 1) 
    #currently, a stroke has a 1 if the next stroke is not drawn
    #the pen pickups are shifted by one, so a stroke that is not drawn has a 1
    strokes[:, :2] /= np.std(strokes[:, :2])
    for i in range(3): strokes = combine_strokes(strokes, int(len(strokes)*0.2))
    return strokes

def read_img(path, height):
    img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')
    img_arr = remove_whitespace(img_arr, thresh=127)
    h, w, _ = img_arr.shape
    img_arr = tf.image.resize(img_arr, (height, height * w // h))
    return img_arr.numpy().astype('uint8')

def create_dataset(formlist, strokes_path, images_path, tokenizer, text_dict, height):
    dataset = []
    offline_dataset = []
    same_writer_examples = []
    forms = open(formlist).readlines()

    for f in forms:
        path = strokes_path + '/' + f[1:4] + '/' + f[1:8]
        offline_path = images_path + '/' + f[1:4] + '/' + f[1:8]

        samples = [s for s in os.listdir(path) if f[1:-1] in s]
        offline_samples = [s for s in os.listdir(offline_path) if f[1:-1] in s]
        shuffled_offline_samples = offline_samples.copy()
        random.shuffle(shuffled_offline_samples)
        
        for i in range(len(samples)):
            dataset.append((
                parse_stroke_xml(path + '/' + samples[i]),
                tokenizer.encode(text_dict[samples[i][:-4]]),
                read_img(offline_path + '/' + shuffled_offline_samples[i], height)
            ))        
    return dataset

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-t', '--text_path', help='path to text labels, \
                        default ./data/ascii-all', default='./data/ascii-all' )

    parser.add_argument('-s', '--strokes_path', help='path to stroke xml, \
                        default ./data/lineStrokes-all', default='./data/lineStrokes-all')

    parser.add_argument('-i', '--images_path', help='path to line images, \
                        default ./data/lineImages-all', default='./data/lineImages-all')
                        
    parser.add_argument('-H', '--height', help='the height of offline images, \
                        default 96', type=int, default= 96)

    args = parser.parse_args()
    t_path = args.text_path
    s_path = args.strokes_path
    i_path = args.images_path
    H = args.height

    train_info = './data/trainset.txt'
    val1_info = './data/testset_f.txt'  #labeled as test, we validation set 1 as test instead
    val2_info = './data/testset_t.txt'  
    test_info = './data/testset_v.txt'  #labeled as validation, but we use as test

    tok = Tokenizer()
    labels = create_dict(t_path)
    train_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    val1_strokes = create_dataset(val1_info, s_path, i_path, tok, labels, H)
    val2_strokes = create_dataset(val2_info, s_path, i_path, tok, labels, H)
    test_strokes = create_dataset(test_info, s_path, i_path, tok, labels, H)
    
    train_strokes += val1_strokes
    train_strokes += val2_strokes
    random.shuffle(train_strokes)
    random.shuffle(test_strokes)

    with open('./data/train_strokes.p', 'wb') as f:
        pickle.dump(train_strokes, f)
    with open('./data/test_strokes.p', 'wb') as f:
        pickle.dump(test_strokes, f)

if __name__ == '__main__':
    main()
