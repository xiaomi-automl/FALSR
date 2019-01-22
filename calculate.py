# encoding: utf-8
import os
import tensorflow as tf
import cv2
import argparse
import numpy as np
import scipy.misc
import imageio
from scipy.misc import imsave
import skimage.color as sc
import skimage
from skimage import measure
import logging
import re
from functools import reduce

scale = 2

def calculate_metrics(hr_y_list, sr_y_list, bnd=2):
    class BaseMetric:
        def __init__(self):
            self.name = 'base'

        def image_preprocess(self, image):
            image_copy = image.copy()
            image_copy[image_copy < 0] = 0
            image_copy[image_copy > 255] = 255
            image_copy = np.around(image_copy).astype(np.double)
            return image_copy

        def evaluate(self, gt, pr):
            pass

        def evaluate_list(self, gtlst, prlst):
            resultlist = list(map(lambda gt, pr: self.evaluate(gt, pr), gtlst, prlst))
            return sum(resultlist) / len(resultlist)


    class PSNRMetric(BaseMetric):
        def __init__(self):
            self.name = 'psnr'

        def evaluate(self, gt, pr):
            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return skimage.measure.compare_psnr(gt, pr, data_range=255)


    class SSIMMetric(BaseMetric):
        def __init__(self):
            self.name = 'ssim'

        def evaluate(self, gt, pr):
            def ssim(img1, img2):
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2

                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                kernel = cv2.getGaussianKernel(11, 1.5)
                window = np.outer(kernel, kernel.transpose())

                mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
                mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
                sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
                sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                return ssim_map.mean()

            def esrgan_ssim(img1, img2):
                if not img1.shape == img2.shape:
                    raise ValueError('Input images must have the same dimensions.')
                if img1.ndim == 2:
                    return ssim(img1, img2)
                elif img1.ndim == 3:
                    if img1.shape[2] == 3:
                        ssims = []
                        for i in range(3):
                            ssims.append(ssim(img1, img2))
                        return np.array(ssims).mean()
                    elif img1.shape[2] == 1:
                        return ssim(np.squeeze(img1), np.squeeze(img2))
                else:
                    raise ValueError('Wrong input image dimensions.')

            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return esrgan_ssim(gt[..., 0], pr[..., 0])


    y_mean_psnr = 0
    y_mean_ssim = 0
    assert len(hr_y_list) == len(sr_y_list)
    for i in range(len(hr_y_list)):
        hr_y, sr_y = hr_y_list[i], sr_y_list[i]
        hr_y = hr_y[bnd:-bnd, bnd:-bnd, :]
        sr_y = sr_y[bnd:-bnd, bnd:-bnd, :]
        y_mean_psnr += PSNRMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
        y_mean_ssim += SSIMMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
    return y_mean_psnr, y_mean_ssim



def exists_or_mkdir(path, verbose=True):
    if not os.path.exists(path):
        if verbose:
            logging.info("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            logging.info("[!] %s exists ..." % path)
        return True



def load_file_list(path=None, regx='\.jpg', printable=True, keep_prefix=False):
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)
    if printable:
        logging.info('Match file list = %s' % return_list)
        logging.info('Number of files = %d' % len(return_list))
    return return_list


def evaluate(calculate_lr_img_list, calculate_hr_img_list, pb_path, save_path, save=False):
    calculate_hr_imgs = [scipy.misc.imread(p, mode='RGB') for p in calculate_hr_img_list]
    calculate_lr_imgs = [scipy.misc.imread(p, mode='RGB') for p in calculate_lr_img_list]


    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            y_image = sess.graph.get_tensor_by_name("input_image_evaluate_y:0")
            pbpr_image = sess.graph.get_tensor_by_name("input_image_evaluate_pbpr:0")
            output_tensor = sess.graph.get_tensor_by_name('test_sr_evaluator_i1_b0_g/target:0')
            sess.run(tf.global_variables_initializer())
            metrics = []
            for index, calculate_lr_img in enumerate(calculate_lr_imgs):
                calculate_hr_img = calculate_hr_imgs[index]
                size = calculate_lr_img.shape
                ypbpr = sc.rgb2ypbpr(calculate_lr_img / 255.0)
                x_scale = scipy.misc.imresize(calculate_lr_img, [size[0] * scale, size[1] * scale], interp='bicubic', mode=None)
                y, pbpr = ypbpr[..., 0], sc.rgb2ypbpr(x_scale / 255)[..., 1:]
                y = np.expand_dims(y, -1)
                paras = {y_image: [y], pbpr_image: [pbpr]}

                out = sess.run(output_tensor, paras)
                out = out[0]

                out = out * 255
                out = np.clip(out, 0, 255)
                out = out.astype(np.uint8)

                if save:
                    exists_or_mkdir(save_path)
                    im = scipy.misc.toimage(out, high=255, low=0)
                    im.save(save_path + os.sep + calculate_hr_img_list[index].split(os.sep)[-1].replace('HR', 'SR'))
                    # imsave(save_path + os.sep + calculate_hr_img_list[index].split(os.sep)[-1].replace('HR', 'SR'), out)
                out_ycbcr = sc.rgb2ycbcr(out)
                hr_ycbcr = sc.rgb2ycbcr(calculate_hr_img)
                metrics.append(calculate_metrics([out_ycbcr[:, :, 0:1]], [hr_ycbcr[:, :, 0:1]]))
            avg_psnr = sum([m[0] for m in metrics])/len(metrics)
            avg_ssim = sum([m[1] for m in metrics])/len(metrics)

            return avg_psnr, avg_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_path", type=str, default='./pretrained_model/FALSR-A.pb')
    parser.add_argument("--save_path", type=str, default='./result/')
    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    calculate_path_list = ['./dataset/Set5', './dataset/Set14', './dataset/B100', './dataset/Urban100']
    for calculate_path in calculate_path_list:
        calculate_lr_img_list = sorted(load_file_list(path=calculate_path, regx='.*LR\.\w+g', printable=False, keep_prefix=True))
        calculate_hr_img_list = sorted(load_file_list(path=calculate_path, regx='.*HR\.\w+g', printable=False, keep_prefix=True))
        print('read %d pairs from %s' % (len(calculate_hr_img_list), calculate_path))

        pb_name = cfg.pb_path.split(os.sep)[-1].split('.')[0]
        save_path = cfg.save_path+os.sep+pb_name+os.sep+calculate_path.split(os.sep)[-1]

        metrics = evaluate(calculate_lr_img_list, calculate_hr_img_list, pb_path=cfg.pb_path, save_path=save_path, save=True)

        print('%s:\n  avg_psnr: %s\n  avg_ssim: %s' % (calculate_path, metrics[0], metrics[1]))

