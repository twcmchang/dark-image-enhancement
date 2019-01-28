# %load PARTIAL-RGB-parametric-sqrt.py
import sys
import os
import tensorflow as tf
import numpy as np
from path import Path
import skimage.io as io
import skimage.transform as transform
import time
import scipy

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return (tf.nn.tanh(x) + 1)/2

def rt0125(x):
    return tf.nn.relu(x) **(1/1.25)

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

group = 4
batch_size =8
target_size = (512,512,3)
n_step = 20000
n_period = 500
decay = 2
save_freq = 50
ssim = False
output_activation = sigmoid
learning_rate = float(sys.argv[2])

data_dir = '/data/put_data/cmchang/dataset/coco_fgbg/'
# pretrain_dir = '/data/put_data/cmchang/partial-pretrain-NNresize-lr0.0001-decay2k-div2/model/'
pretrain_dir = ''

if pretrain_dir:
    addstr = 'pretrain-'
else:
    addstr = ''

RES_DIR = '/data/put_data/cmchang/value-group{}-rgb-parametric-{}-Truncated-lowVar-center0-lr{}-decay{}k-div{}-ssim{}/'.format(group, output_activation.__name__, learning_rate, n_period/1000, decay, int(ssim))
checkpoint_dir = os.path.join(RES_DIR, 'model/')
result_dir = os.path.join(RES_DIR, 'image/')

if not os.path.exists(RES_DIR):
    os.mkdir(RES_DIR)
    
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    
def lrelu(x):
    return tf.maximum(x * 0.2, x)

def linear(x):
    return x

def sqrt(x):
    return tf.sqrt(x)

def upsample_and_concat(x1, x2, output_channels, name):
    inshape = x1.get_shape().as_list()
    outshape = x2.get_shape().as_list()

    H, W = outshape[1], outshape[2]
    conv = tf.image.resize_nearest_neighbor(x1, [H,W])

    deconv = tf.layers.conv2d(conv, output_channels, [3, 3], activation=lrelu, padding='SAME', name=name)

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output

def group_img_pixel(input_image, n_group):
    """
    @input_image: rescale to [0, 1]
    @n_group: equally separated by n_group
        - example: n_group = 2, [0, 0.5), and [0.5, 1)
    """
    #gray_image = tf.reduce_sum(input_image*[0.299, 0.587, 0.114], axis=3)
    gray_image = tf.reduce_sum(input_image, axis=3)
    gray_image = tf.stack([gray_image]*3, axis=3)
    
    intv = 1.0 / n_group
    temp_input_list = list()
    mask_list = list()
    for i in range(0, n_group):
        t = i * intv
        if i == (n_group-1):
            mask = tf.cast(gray_image>=t, dtype=tf.float32)
        else:
            mask = tf.cast(tf.logical_and(gray_image >= t, gray_image < (t+intv)), dtype=tf.float32)
        mask_list.append(mask)
        temp_input_list.append(tf.multiply(input_image, mask))
        
    final_input = tf.concat(temp_input_list, axis=3)
    mask_input = tf.concat(mask_list, axis=3)
    
    return final_input, mask_input

def network(input_image, n_group=4):
    #shape_list.append(inputs.get_shape())
    
    inputs, masks = group_img_pixel(input_image, n_group)
    
    conv1 = tf.layers.conv2d(inputs, 32, [3, 3], activation=lrelu, padding='SAME', name='g_conv1_1')
    conv1 = tf.layers.conv2d(conv1, 32, [3, 3], activation=lrelu, padding='SAME', name='g_conv1_2')
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='SAME')
    
    #shape_list.append(pool1.get_shape())

    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], activation=lrelu, padding='SAME', name='g_conv2_1')
    conv2 = tf.layers.conv2d(conv2, 64, [3, 3], activation=lrelu, padding='SAME', name='g_conv2_2')
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME')

    #shape_list.append(pool2.get_shape())
    
    conv3 = tf.layers.conv2d(pool2, 128, [3, 3], activation=lrelu, padding='SAME', name='g_conv3_1')
    conv3 = tf.layers.conv2d(conv3, 128, [3, 3], activation=lrelu, padding='SAME', name='g_conv3_2')
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='SAME')
    
    #shape_list.append(pool3.get_shape())

    conv4 = tf.layers.conv2d(pool3, 256, [3, 3], activation=lrelu, padding='SAME', name='g_conv4_1')
    conv4 = tf.layers.conv2d(conv4, 256, [3, 3], activation=lrelu, padding='SAME', name='g_conv4_2')
    pool4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='SAME')

    #shape_list.append(pool4.get_shape())
    
    conv5 = tf.layers.conv2d(pool4, 512, [3, 3], activation=lrelu, padding='SAME', name='g_conv5_1')
    conv5 = tf.layers.conv2d(conv5, 512, [3, 3], activation=lrelu, padding='SAME', name='g_conv5_2')
    
    global_avg = tf.reduce_mean(conv5, axis=[1,2])
    
    # out[0]: Brightness, out[1]: Contrast
    a = tf.layers.dense(global_avg, activation=linear, units=3*n_group, name='scale')
    b = tf.layers.dense(global_avg, activation=linear, units=3*n_group, name='shift')
    
    new_image = tf.map_fn(lambda x: (x[0] * x[1] + x[2])*x[3], (inputs, a, b, masks), dtype=tf.float32)
    new_image_R = tf.reduce_sum(tf.gather(new_image, indices=list(range(0, 3*n_group, 3)), axis=3), axis=3)
    new_image_G = tf.reduce_sum(tf.gather(new_image, indices=list(range(1, 3*n_group, 3)), axis=3), axis=3)
    new_image_B = tf.reduce_sum(tf.gather(new_image, indices=list(range(2, 3*n_group, 3)), axis=3), axis=3)
    new_image_final = tf.stack([new_image_R, new_image_G, new_image_B], axis=3)
    out_image = output_activation(new_image_final)
#     out_image = tf.nn.relu(new_image_final)**(1/1.25)
    
    return out_image, a, b

class DataGenerator():
    def __init__(self, data_dir, batch_size=32, dim=(512,512,3)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        self.bg_IDs, self.fg_IDs = self.__list_all_images(data_dir)
        
        assert len(self.bg_IDs) == len(self.fg_IDs)
        self.indexes = np.arange(len(self.bg_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.bg_IDs)
    
    def __list_all_images(self, args_dir):
        # give priority to directory
        imgs = list()
        if args_dir is not None:
            print("Loading images from directory : ", args_dir)
            dir_list = [Path(args_dir)]
            while(dir_list):
                d = dir_list.pop()
                if d.dirs():
                    dir_list.extend(d.dirs())
                imgs += d.files('*.png')
                imgs += d.files('*.jpg')
                imgs += d.files('*.jpeg')
                imgs += d.files('*.bmp')
                imgs += d.files('*.BMP')
        bg_imgs, fg_imgs = list(), list()
        imgs = sorted(imgs)
        for img in imgs:
            if 'bg' in img.name:
                bg_imgs.append(img)
            if 'fg' in img.name:
                fg_imgs.append(img)
        return bg_imgs, fg_imgs
    
    def remove_image_threshold(self, threshold):
        n_original = len(self.bg_IDs)
        self.bg_IDs = [img for img in self.bg_IDs if float(img.name[:-4].split('_')[1]) > threshold]
        print('remain {} bg images'.format(len(self.bg_IDs)))
        
        n_original = len(self.fg_IDs)
        self.fg_IDs = [img for img in self.fg_IDs if float(img.name[:-4].split('_')[1]) > threshold]
        print('remain {} fg images'.format(len(self.fg_IDs)))
        
        self.indexes = np.arange(len(self.bg_IDs))

    def next_batch(self, random=True):
        'Generate one batch of data'
        # Generate indexes of the batch
        if random:
            sel = np.random.randint(0, len(self.bg_IDs), self.batch_size)
        else:
            sel = np.arange(self.batch_size)
        indexes = self.indexes[sel]

        # Find list of IDs
        list_IDs_temp = [self.bg_IDs[k] for k in indexes]

        # Generate data
        BG = self.__data_generation(list_IDs_temp)
        
        list_IDs_temp = [self.fg_IDs[k] for k in indexes]
        FG = self.__data_generation(list_IDs_temp)

        return FG, BG

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img_list = list()
        for f in list_IDs_temp:
            a = io.imread(f)
            a = transform.resize(a, output_shape= self.dim)
            # a = (a*2.0)-1.0 # normalize to [-1,1]
            img_list.append(a) 
        return np.array(img_list)

def directly_read_all_images(args_dir, target_size):
    imgs = list()
    # give priority to directory
    if args_dir is not None:
        print("Loading images from directory : ", args_dir)
        dir_list = [Path(args_dir)]
        dir_list.extend(Path(args_dir).dirs())
        while(dir_list):
            d = dir_list.pop()
            if d.dirs():
                dir_list.extend(d.dirs())
            imgs += d.files('*.png')
            imgs += d.files('*.jpg')
            imgs += d.files('*.jpeg')
            imgs += d.files('*.bmp')
            imgs += d.files('*.BMP')
            imgs += d.files('*.JPG')
            imgs += d.files('*.JPEG')
    img_list = list()
    for f in imgs:
        a = io.imread(f)
        a = transform.resize(a, output_shape=target_size)
        # a = (a*2.0)-1.0 # rescale to [-1, 1]
        img_list.append(a)
    return np.array(img_list)

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): 
            sess.run(tf.variables_initializer(not_initialized_vars))


### MAIN ###

dgen = DataGenerator('/data/put_data/cmchang/dataset/coco_fgbg/', batch_size=batch_size, dim=target_size)
dgen.remove_image_threshold(0.05)

Xtest_fg, Xtest_bg = dgen.next_batch(random=False)

with tf.variable_scope('unet'):
    
    # 3 inputs: foreground images, background images, mean
    fore_image = tf.placeholder(shape=[batch_size, *target_size], dtype=tf.float32)
    back_image = tf.placeholder(shape=[batch_size, *target_size], dtype=tf.float32)
    mean = tf.placeholder(dtype=tf.float32)
    
    # original image as label
    gt_image = tf.clip_by_value(fore_image + back_image, clip_value_max=1.0, clip_value_min=0.0)
    
    # reduce contrast -> surpress foreground and background using different normal distribution
    factor = np.clip(np.random.normal(0.85, 0.05), a_max=1.0, a_min=0.7)
    
    fore_darker = tf.minimum(0.8*mean+0.1, tf.maximum(0.02, tf.random_normal([fore_image.get_shape().as_list()[0]], 0.8*mean, 0.1)))#-0.6*mean, 0.02)) # 0.8
    fore_image_transpose = tf.transpose(fore_image, [1,2,3,0])
    # fore_darker = fore_dark * tf.constant([0.229*3, 0.587*3, 0.114*3])

    # fore_image_dark = tf.image.adjust_brightness(fore_image_low_constrast, fore_darker) 
    fore_image_dark = fore_image_transpose + fore_darker * ([-1, 1] * (batch_size // 2))

    fore_image_dark = tf.transpose(fore_image_dark, [3,0,1,2])

    fore_image_low_contrast = fore_image_dark * factor + mean * (1.-factor)

    
    back_darker = tf.minimum(0.4*mean+0.05, tf.maximum(0.01, tf.random_normal([fore_image.get_shape().as_list()[0]], 0.4*mean, 0.05)) )#-0.3*mean, 0.02)) # 0.4
    # back_darker = back_dark * tf.constant([0.229*3, 0.587*3, 0.114*3])
    # back_image_dark = tf.image.adjust_brightness(back_image_low_constrast, back_darker)         
    back_image_transpose = tf.transpose(back_image, [1,2,3,0])
    back_image_dark = back_image_transpose + back_darker * ([-1, 1] * (batch_size // 2))
    back_image_dark = tf.transpose(back_image_dark, [3,0,1,2])

    back_image_low_contrast = back_image_dark * factor + mean * (1.-factor)

    in_image = tf.clip_by_value(fore_image_low_contrast + back_image_low_contrast, clip_value_max=1.0, clip_value_min=0.0)
    in_image.set_shape([None, *target_size])
    
    out_image, a, b = network(in_image, n_group=group)
    
    # out_image = tf.nn.sigmoid(new_image)

    img_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
 
    # if ssim:
    #     ms_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim_multiscale(gt_image, out_image, max_val = 1.0))
    #     G_loss = img_loss + ms_ssim_loss
    # else:
    G_loss = img_loss

    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32)
    # G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

    with tf.Session() as sess:
        best_loss = np.float('Inf')
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(pretrain_dir)

        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
        initialize_uninitialized(sess)

        loss_list = list()
        
        for ind in range(n_step):
            st = time.time()
            if ind % n_period == 0 and ind > 0:
                learning_rate = learning_rate / 2
                n_period = n_period * 2

            fg, bg = dgen.next_batch()

            _, G_current = sess.run([G_opt, G_loss],
                                    feed_dict={fore_image: fg, back_image: bg, lr: learning_rate, mean: np.mean(fg+bg)})

            loss_list.append(G_current)

            print("step %05d Loss=%.3f Time=%.3f" % (ind, G_current, time.time() - st))

            if ind % save_freq == 0:
                inputs, outputs, originals = sess.run([in_image, out_image, gt_image], feed_dict={fore_image: Xtest_fg, 
                                                                             back_image: Xtest_bg, 
                                                                             mean: np.mean(Xtest_fg+Xtest_bg)})
                for ni in range(outputs.shape[0]):
                    temp = np.concatenate((inputs[ni, :, :, :], outputs[ni, :, :, :], originals[ni,:,:,:]), axis=1)
                    # temp = (temp + 1)/2
                    scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                        result_dir + '%05d_train_%d.jpg' % (ind, ni))
            with open(checkpoint_dir + 'log.txt', 'w') as f:
                f.write('epoch,loss\n')
                f.write(''.join(['{},{:4f}\n'.format(i,ls) for i, ls in enumerate(loss_list)]))        
            if G_current < best_loss:
                best_loss = G_current
                saver.save(sess, checkpoint_dir + 'model.ckpt')
