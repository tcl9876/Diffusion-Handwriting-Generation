import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, Conv1D, Embedding, UpSampling1D, AveragePooling1D, 
AveragePooling2D, GlobalAveragePooling2D, Activation, LayerNormalization, Dropout, Layer)

def create_padding_mask(seq, repeats=1):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.repeat(seq, repeats=repeats, axis=-1)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    return mask

def get_angles(pos, i, C, pos_factor = 1):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(C))
    return pos * angle_rates * pos_factor

def positional_encoding(position, C, pos_factor=1):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(C)[np.newaxis, :], C, pos_factor=pos_factor)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])    
    pos_encoding = angle_rads[np.newaxis, ...]    
    return tf.cast(pos_encoding, dtype=tf.float32)
    
def ff_network(C, dff=768, act_before=True):
    ff_layers = [
        Dense(dff, activation='swish'),
        Dense(C)    
    ]
    if act_before: ff_layers.insert(0, Activation('swish'))
    return Sequential(ff_layers)   
    
def loss_fn(eps, score_pred, pl, pl_pred, abar, bce):
    score_loss = tf.reduce_mean(tf.reduce_sum(tf.square(eps - score_pred), axis=-1)) 
    pl_loss = tf.reduce_mean(bce(pl, pl_pred) * tf.squeeze(abar, -1))
    return score_loss + pl_loss
    
def scaled_dp_attn(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True) #batch_size, d_model, seq_len_q, seq_len_k
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  
    scaled_qk = qk / tf.sqrt(dk)
    if mask is not None: scaled_qk += (mask*-1e12)
    
    attention_weights = tf.nn.softmax(scaled_qk, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

def reshape_up(x, factor=2):
    x_shape = tf.shape(x)
    x = tf.reshape(x, [x_shape[0], x_shape[1]*factor, x_shape[2]//factor])
    return x

def reshape_down(x, factor=2):
    x_shape = tf.shape(x)
    x = tf.reshape(x, [x_shape[0], x_shape[1]//factor, x_shape[2]*factor])
    return x
    
class AffineTransformLayer(Layer):
    def __init__(self, filters):
        super().__init__()
        self.gamma_emb = Dense(filters, bias_initializer='ones')
        self.beta_emb = Dense(filters)
    
    def call(self, x, sigma):
        gammas = self.gamma_emb(sigma)
        betas = self.beta_emb(sigma)
        return x * gammas + betas

class MultiHeadAttention(Layer):
    def __init__(self, C, num_heads):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.wq = Dense(C)
        self.wk = Dense(C)
        self.wv = Dense(C)
        self.dense = Dense(C)  
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.C // self.num_heads))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v) # (bs, sl, C)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size) #(bs, nh, sl, C // nh) for q,k,v

        attention, attention_weights = scaled_dp_attn(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (bs, sl, nh, C // nh)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.C)) # (bs, sl, c)
        output = self.dense(concat_attention)
        return output, attention_weights

class InvSqrtSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class AffineTransformLayer(Layer):
    def __init__(self, filters):
        super().__init__()
        self.gamma_dense = Dense(filters, bias_initializer='ones')
        self.beta_dense = Dense(filters)
    
    def call(self, x, sigma):
        gammas = self.gamma_dense(sigma)
        betas = self.beta_dense(sigma)
        return x * gammas + betas

class ConvSubLayer(Model):
    def __init__(self, filters, dils=[1,1], activation='swish', drop_rate=0.0):
        super().__init__()
        self.act = Activation(activation)
        self.affine1 = AffineTransformLayer(filters//2)
        self.affine2 = AffineTransformLayer(filters)
        self.affine3 = AffineTransformLayer(filters)
        self.conv_skip = Conv1D(filters, 3, padding='same')
        self.conv1 = Conv1D(filters//2, 3, dilation_rate=dils[0], padding='same')
        self.conv2 = Conv1D(filters, 3, dilation_rate=dils[0], padding='same')
        self.fc = Dense(filters)
        self.drop = Dropout(drop_rate)

    def call(self, x, alpha):
        x_skip = self.conv_skip(x)
        x = self.conv1(self.act(x))
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.act(x))
        x = self.drop(self.affine2(x, alpha))
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))
        x += x_skip
        return x

class StyleExtractor(Model):
    #takes a grayscale image (with the last channel) with pixels [0, 255]
    #rescales to [-1, 1] and repeats along the channel axis for 3 channels
    #uses a MobileNetV2 with pretrained weights from imagenet as initial weights

    def __init__(self):
        super().__init__()
        self.mobilenet = MobileNetV2(include_top=False, pooling=None, weights='imagenet', input_shape=(96, 96, 3))
        self.local_pool = AveragePooling2D((3,3))
        self.global_avg_pool = GlobalAveragePooling2D()
        self.freeze_all_layers()

    def freeze_all_layers(self,):
        for l in self.mobilenet.layers:
            l.trainable = False
    
    def call(self, im, im2=None, get_similarity=False, training=False):
        x = tf.cast(im, tf.float32)
        x = (x / 127.5) - 1     
        x = tf.repeat(x, 3, axis=-1)

        x = self.mobilenet(x, training=training)
        x = self.local_pool(x)
        output = tf.squeeze(x, axis=1)
        return output
        
class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, drop_rate=0.1, pos_factor=1):
        super().__init__()
        self.text_pe = positional_encoding(2000, d_model, pos_factor=1)
        self.stroke_pe = positional_encoding(2000, d_model, pos_factor=pos_factor)
        self.drop = Dropout(drop_rate)
        self.lnorm = LayerNormalization(epsilon=1e-6, trainable=False)
        self.text_dense = Dense(d_model)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = ff_network(d_model, d_model*2)
        self.affine0 = AffineTransformLayer(d_model)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
    
    def call(self, x, text, sigma, text_mask):
        text = self.text_dense(tf.nn.swish(text))
        text = self.affine0(self.lnorm(text), sigma)
        text_pe = text + self.text_pe[:, :tf.shape(text)[1]]

        x_pe = x + self.stroke_pe[:, :tf.shape(x)[1]]
        x2, att = self.mha(x_pe, text_pe, text, text_mask)
        x2 = self.lnorm(self.drop(x2))
        x2 = self.affine1(x2, sigma) + x

        x2_pe = x2 + self.stroke_pe[:, :tf.shape(x)[1]]
        x3, _ = self.mha2(x2_pe, x2_pe, x2)
        x3 = self.lnorm(x2 + self.drop(x3))
        x3 = self.affine2(x3, sigma)

        x4 = self.ffn(x3)
        x4 = self.drop(x4) + x3
        out = self.affine3(self.lnorm(x4), sigma)
        return out, att

class Text_Style_Encoder(Model):
    def __init__(self, d_model, d_ff=512):
        super().__init__()
        self.emb = Embedding(73, d_model)
        self.text_conv = Conv1D(d_model, 3, padding='same')
        self.style_ffn = ff_network(d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, 8)
        self.layernorm = LayerNormalization(epsilon=1e-6, trainable=False)
        self.dropout = Dropout(0.3)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_ffn = ff_network(d_model, d_model*2)

    def call(self, text, style, sigma):
        style = reshape_up(self.dropout(style), 5)
        style = self.affine1(self.layernorm(self.style_ffn(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layernorm(text), sigma)
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layernorm(text + mha_out), sigma)
        text_out = self.affine4(self.layernorm(self.text_ffn(text)), sigma)
        return text_out

class DiffusionWriter(Model):
    def __init__(self, num_layers=4, c1=128, c2=192, c3=256, drop_rate=0.1, num_heads=8):
        super().__init__()
        self.input_dense = Dense(c1)
        self.sigma_ffn = ff_network(c1//4, 2048)
        self.enc1 = ConvSubLayer(c1, [1, 2])
        self.enc2 = ConvSubLayer(c2, [1, 2])
        self.enc3 = DecoderLayer(c2, 3, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c3, [1, 2])
        self.enc5 = DecoderLayer(c3, 4, drop_rate, pos_factor=2)
        self.pool = AveragePooling1D(2)
        self.upsample = UpSampling1D(2)

        self.skip_conv1 = Conv1D(c2, 3, padding='same')
        self.skip_conv2 = Conv1D(c3, 3, padding='same')
        self.skip_conv3 = Conv1D(c2*2, 3, padding='same')
        self.text_style_encoder = Text_Style_Encoder(c2*2, c2*4)
        self.att_dense = Dense(c2*2)
        self.att_layers = [DecoderLayer(c2*2, 6, drop_rate) 
                     for i in range(num_layers)]
                     
        self.dec3 = ConvSubLayer(c3, [1, 2])
        self.dec2 = ConvSubLayer(c2, [1, 1])
        self.dec1 = ConvSubLayer(c1, [1, 1])
        self.output_dense = Dense(2)
        self.pen_lifts_dense = Dense(1, activation='sigmoid')
 
    def call(self, strokes, text, sigma, style_vector):
        sigma = self.sigma_ffn(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        x = self.input_dense(strokes)
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)
        h2, _ = self.enc3(h2, text, sigma, text_mask)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        x = self.pool(h3)
        
        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)

        x = self.upsample(x) + self.skip_conv3(h3)
        x = self.dec3(x, sigma)

        x = self.upsample(x) + self.skip_conv2(h2)
        x = self.dec2(x, sigma)

        x = self.upsample(x) + self.skip_conv1(h1)
        x = self.dec1(x, sigma)
        
        output = self.output_dense(x)
        pl = self.pen_lifts_dense(x)
        return output, pl, att