from __future__ import print_function
import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict

sys.path.append('./')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2

# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr
from scipy.spatial import KDTree

#
import argparse
import torch.nn.functional as F

label_colours = [(0, 0, 0)
    , (128, 0, 0), (255, 0, 0), (0, 85, 0), (170, 0, 51), (255, 85, 0), (0, 0, 85), (0, 119, 221), (85, 85, 0),
                 (0, 85, 85), (85, 51, 0), (52, 86, 128), (0, 128, 0)
    , (0, 0, 255), (51, 170, 221), (0, 255, 255), (85, 255, 170), (170, 255, 85), (255, 255, 0), (255, 170, 0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev, dim=0)

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def get_nearest_simple_color_rgb(rgb):
  names = ['Red', 'Red', 'White', 'Black', 'Green', 'Blue', 'Yellow', 'Aqua', 'Magenta', 'Brown', 'Gray']
  positions = [(255,0,0), (128, 0, 0), (255,255,255), (0,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (170,110,140), (128,128,128)]
  spacedb = KDTree(positions)
  querycolor = rgb
  dist, index = spacedb.query(querycolor)
  # print('The color %r is closest to %s.'%(querycolor, names[index]))
  return positions[index], names[index]
  # print('The color %r is closest to %s.'%(querycolor, names[index]))


def dominant_color(pixels, img_path):
  nearest_colors_list = list()
  im = Image.open(img_path)
  pix = im.load()
  for i in pixels:
      actual_color = pix[i[0],i[1]]
      nearest_colors_list.append(get_nearest_simple_color_rgb(actual_color)[0])
  freq = {}
  for item in nearest_colors_list:
      if (item in freq):
          freq[item] += 1
      else:
          freq[item] = 1
  print(freq)
  freq = sorted(freq, key=freq.get, reverse=True)
  return freq[0], freq[1], freq[2]


def decode_labels(mask, img_path, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)

    pixels_with_shirt = list()
    im = Image.open(
        img_path)  # Can be many different formats.
    pix = im.load()

    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
                    if (k in [5, 6, 7, 10]):
                        pixels_with_shirt.append([k_, j_])
        outputs[i] = np.array(img)
    # print([sum(ele) / len(color_of_pixels_with_shirt) for ele in zip(*color_of_pixels_with_shirt)])
    # dominant_color_rgb = dominant_color(pixels_with_shirt, img_path)
    # print("dominant_color_rgb", dominant_color_rgb)
    # print(get_nearest_simple_color_rgb(dominant_color_rgb[0]))
    # print(get_nearest_simple_color_rgb(dominant_color_rgb[1]))
    # print(get_nearest_simple_color_rgb(dominant_color_rgb[2]))
    #
    # coords = (bounding_box(pixels_with_shirt))
    # # print(coords)
    # print_pic(coords, img_path, dominant_color_rgb)
    return outputs

def print_pic(coords, img_path, colors):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  from PIL import Image

  im = Image.open(img_path)

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  rect = patches.Rectangle(coords[0], coords[1][0] - coords[0][0], coords[1][1] - coords[0][1], linewidth=1, edgecolor='r', facecolor='none')

  # Add the patch to the Axes
  ax.add_patch(rect)
  colors = [get_nearest_simple_color_rgb(x)[1] for x in colors]
  ax.annotate(str(colors), coords[0], color='black', weight='bold', fontsize=10, ha='center', va='center')
  plt.savefig(img_path[:-3]+'new.png' ,dpi=300, bbox_inches = "tight")
  plt.show()

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img


def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample


def inference(net, img_path='', output_path='./', output_name='f', use_gpu=True):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img = read_img(img_path)
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)
            # print((outputs.size()))
            # print(outputs)
            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    # print("Size", torch.Size(predictions))
    # print("pred", predictions)
    results = predictions.cpu().numpy()
    # print("Results", results)
    # print("Size", results.shape)
    # print("results", results)
    vis_res = decode_labels(results, img_path)
    # print("vis_res", vis_res)

    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(output_path + '/{}.png'.format(output_name))
    # cv2.imwrite(output_path + '/{}_gray.png'.format(output_name), results[0, :, :])

    end_time = timeit.default_timer()
    print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))


if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--img_dir', default='', type=str)
    parser.add_argument('--output_path', default='', type=str)
    # parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--use_gpu', default=1, type=int)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    if not opts.loadmodel == '':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:', opts.loadmodel)
    else:
        print('no model load !!!!!!!!')
        raise RuntimeError('No model!!!!')

    if opts.use_gpu > 0:
        net.cuda()
        use_gpu = True
    else:
        use_gpu = False
        raise RuntimeError('must use the gpu!!!!')

    import os
    from os import listdir
    from os.path import isfile, join

    onlyfiles = [f for f in listdir(opts.img_dir) if isfile(join(opts.img_dir, f))]
    for file in onlyfiles:
        inference(net=net, img_path=opts.img_dir + '/' + file, output_path=opts.output_path, output_name=file + '_out',
                  use_gpu=use_gpu)
