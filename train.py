import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import nn
import time
import argparse

@tf.function
def train_step(x, pen_lifts, text, style_vectors, glob_args):
    model, alpha_set, bce, train_loss, optimizer = glob_args
    alphas = utils.get_alphas(len(x), alpha_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alphas) * x 
    x_perturbed += tf.sqrt(1 - alphas) * eps
    
    with tf.GradientTape() as tape:
        score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
        loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
        
    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return score, att

def train(dataset, iterations, model, optimizer, alpha_set, print_every=1000, save_every=10000):
    s = time.time()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean()
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        glob_args = model, alpha_set, bce, train_loss, optimizer
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors, glob_args)
        
        if optimizer.iterations%print_every==0:
            print("Iteration %d, Loss %f, Time %ds" % (optimizer.iterations, train_loss.result(), time.time()-s))
            train_loss.reset_states()

        if (optimizer.iterations+1) % save_every==0:
            save_path = ckpt_path + './weights/model_step%d.h5' % (optimizer.iterations+1)
            model.save_weights(save_path)
            
        if optimizer.iterations > iterations:
            model.save_weights('./weights/model.h5')
            break

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--steps', help='number of trainsteps, default 60k', default=60000, type=int)
    parser.add_argument('--batchsize', help='default 96', default=96, type=int)
    parser.add_argument('--seqlen', help='sequence length during training, default 480', default=480, type=int)
    parser.add_argument('--textlen', help='text length during training, default 50', default=50, type=int)
    parser.add_argument('--width', help='offline image width, default 1400', default=1400, type=int)
    parser.add_argument('--warmup', help='number of warmup steps, default 10k', default=10000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--print_every', help='show train loss every n iters', default=1000, type=int)
    parser.add_argument('--save_every', help='save ckpt every n iters', default=10000, type=int)

    args = parser.parse_args()
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batchsize
    MAX_SEQ_LEN = args.seqlen
    MAX_TEXT_LEN = args.textlen
    WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    SAVE_EVERY = args.save_every
    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN%8) + 8

    BUFFER_SIZE = 3000
    L = 60
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    alpha_set = tf.math.cumprod(1-beta_set)

    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE)
    lr = nn.InvSqrtSchedule(C3, warmup_steps=WARMUP_STEPS)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, clipnorm=100)
    
    path = './data/train_strokes.p'
    strokes, texts, samples = utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, 96)
    dataset = utils.create_dataset(strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE)

    train(dataset, NUM_STEPS, model, optimizer, alpha_set, PRINT_EVERY, SAVE_EVERY)

if __name__ == '__main__':
    main()
