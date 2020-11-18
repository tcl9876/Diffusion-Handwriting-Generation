import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import string
import pickle
import os

#notation clarification:
#we use the variable "alpha" for alpha_bar (cumprod 1-beta)
#the alpha in the paper is replaced with 1-beta
def explin(min, max, L):
    return tf.exp(tf.linspace(tf.math.log(min), tf.math.log(max), L))

def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set
    
def show(strokes, name='', show_output=True, scale=1):
    positions = np.cumsum(strokes, axis=0).T[:2]
    prev_ind = 0
    W, H = np.max(positions, axis=-1) - np.min(positions, axis=-1)
    plt.figure(figsize=(scale * W/H, scale))

    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5: 
            plt.plot(positions[0][prev_ind:ind], positions[1][prev_ind:ind], color='black')
            prev_ind = ind
        
    plt.axis('off')
    if name: plt.savefig('./' + name + '.png', bbox_inches='tight')
    if show_output:  plt.show()
    else: plt.close()
    
def get_alphas(batch_size, alpha_set): 
    alpha_indices = tf.random.uniform([batch_size, 1], maxval=len(alpha_set) - 1, dtype=tf.int32)
    lower_alphas = tf.gather_nd(alpha_set, alpha_indices)
    upper_alphas = tf.gather_nd(alpha_set, alpha_indices+1)
    alphas = tf.random.uniform(lower_alphas.shape, maxval=1) * (upper_alphas - lower_alphas) 
    alphas += lower_alphas
    alphas = tf.reshape(alphas, [batch_size, 1, 1])
    return alphas

def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    x_t_minus1 = (1 / tf.sqrt(1-beta)) * (xt - (beta * eps/tf.sqrt(1-alpha)))
    if add_sigma: x_t_minus1 += tf.sqrt(beta) * (tf.random.normal(xt.shape))
    return x_t_minus1
   
def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    x_t_minus1 = (xt - tf.sqrt(1-alpha)*eps) / tf.sqrt(1-beta)
    x_t_minus1 += tf.random.normal(xt.shape) * tf.sqrt(1-alpha_next)
    return x_t_minus1
    
def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new', show_every=None, show_samples=True, path=None):
    if isinstance(text, str):
        text = tf.constant([tokenizer.encode(text)+[1]])
    elif isinstance(text, list) and isinstance(text[0], str):
        tmp = []
        for i in text:
            tmp.append(tokenizer.encode(i)+[1])
        text = tf.constant(tmp)

    bs = text.shape[0]
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1- beta_set)
    x = tf.random.normal([bs, time_steps, 2])
    
    for i in range(L-1, -1, -1):
        alpha = alpha_set[i] * tf.ones([bs, 1, 1]) 
        beta = beta_set[i] * tf.ones([bs, 1, 1]) 
        a_next = alpha_set[i-1] if i>1 else 1.
        model_out, pen_lifts, att = model(x, text, tf.sqrt(alpha), style)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i)) 
        else: 
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)
        
        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0])
                plt.show()

    x = tf.concat([x, pen_lifts], axis=-1)
    for i in range(bs):
        show(x[i], scale=1, show_output = show_samples, name=path)

    return x.numpy()
    
def pad_stroke_seq(x, maxlength):
    if len(x) > maxlength or np.amax(np.abs(x)) > 15: return None
    zeros = np.zeros((maxlength - len(x), 2))
    ones = np.ones((maxlength - len(x), 1))
    padding = np.concatenate((zeros, ones), axis=-1)
    x = np.concatenate((x, padding)).astype('float32')
    return x

def pad_img(img, width, height):
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img
	
def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
        
    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len-len(text), ))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            if x is not None and sample.shape[1] < img_width: 
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples
    
def create_dataset(strokes, texts, samples, style_extractor, batch_size, buffer_size):    
    #we DO NOT SHUFFLE here, because we will shuffle later
    samples = tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
    for count, s in enumerate(samples):
        style_vec = style_extractor(s)
        style_vec = style_vec.numpy()
        if count==0: style_vectors = np.zeros((0, style_vec.shape[1], 1280))
        style_vectors = np.concatenate((style_vectors, style_vec), axis=0)
    style_vectors = style_vectors.astype('float32')
    
    dataset = tf.data.Dataset.from_tensor_slices((strokes, texts, style_vectors))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    
class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = '_' + string.ascii_letters + string.digits + '.?!,\'\"- '
        self.numbers = np.arange(2, len(self.text)+2)
        self.create_dict()
        self.vocab_size = len(self.text)+2
    
    def create_dict(self):
        for char, token, in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = ' ', '<end>' #only for decoding

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text: tokenized.append(self.tokens[char])
            else: tokenized.append(2) #unknown character is '_', which has index 2
         
        tokenized.append(1) #1 is the end of sentence character
        return tokenized
    
    def decode(self, tokens):
        if isinstance(tokens, tf.Tensor): tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return ''.join(text)
