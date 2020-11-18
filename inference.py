import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import nn
import argparse
import os
import preprocessing

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--textstring', help='the text you want to generate', default='Generating text', type=str)  
    parser.add_argument('--writersource', help="path of the image of the desired writer, (e.g. './assets/image.png'   \
                                                will use random from ./assets if unspecified", default=None)
    parser.add_argument('--name', help="path for generated image (e.g. './assets/sample.png'), \
                                             will not be saved if unspecified", default=None)
    parser.add_argument('--diffmode', help="what kind of y_t-1 prediction to use, use 'standard' for  \
                                            Eq 9 in paper, will default to prediction in Eq 12", default='new', type=str)
    parser.add_argument('--show', help="whether to show the sample (popup from matplotlib)", default=False, type=bool)
    parser.add_argument('--weights', help='the path of the loaded weights', default='./weights/model_weights.h5', type=str)
    parser.add_argument('--seqlen', help='number of timesteps in generated sequence, default 16 * length of text', default=None, type=int)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution, \
                                                 only change this if loaded model was trained with that hyperparameter', default=2, type=int)
    parser.add_argument('--channels', help='number of channels at lowest resolution, only change \
                                                 this if loaded model was trained with that hyperparameter', default=128, type=int)
    
    args = parser.parse_args()
    timesteps = len(args.textstring) * 16 if args.seqlen is None else args.seqlen
    timesteps = timesteps - (timesteps%8) + 8 
    #must be divisible by 8 due to downsampling layers

    if args.writersource is None:
        assetdir = os.listdir('./assets')
        sourcename = './assets/' + assetdir[np.random.randint(0, len(assetdir))]
    else: 
        sourcename = args.writersource
 
    L = 60
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    alpha_set = tf.math.cumprod(1-beta_set)

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3)
    
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    #we have to call the model on input first
    model.load_weights(args.weights)

    writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
    style_vector = style_extractor(writer_img)
    utils.run_batch_inference(model, beta_set, args.textstring, style_vector, 
                                tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args.diffmode, 
                                show_samples=args.show, path=args.name)

if __name__ == '__main__':
    main()