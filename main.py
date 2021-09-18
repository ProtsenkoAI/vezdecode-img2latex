from dataset.dataset import test_transform
import cv2
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
import sys
import argparse
import logging
import yaml
import re

import numpy as np
import torch
from torchvision import transforms
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

import warnings

from dataset.latex2png import tex2pil
from models import get_model
from utils import *

last_pic = None

warnings.simplefilter(action="ignore")


def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def initialize(arguments=None, katex=False):
    if arguments is None:
        arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False, "katex": katex})
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = Munch(params)
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if 'image_resizer.pth' in os.listdir(os.path.dirname(args.checkpoint)) and not arguments.no_resize:
        image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=22, global_pool='avg', in_chans=1, drop_rate=.05,
                                 preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args.device)
        image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(args.checkpoint), 'image_resizer.pth'), map_location=args.device))
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, image_resizer, tokenizer


def call_model(args, model, image_resizer, tokenizer, img=None):
    global last_pic
    encoder, decoder = model.encoder, model.decoder
    if type(img) is bool:
        img = None
    if img is None:
        if last_pic is None:
            print('Provide an image.')
            return
        else:
            img = last_pic.copy()
    else:
        last_pic = img.copy()
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    if image_resizer is not None and not args.no_resize:
        with torch.no_grad():
            input_image = pad(img).convert('RGB').copy()
            r, w = 1, img.size[0]
            for i in range(10):
                img = minmax_size(input_image.resize((w, int(input_image.size[1]*r)), Image.BILINEAR if r > 1 else Image.LANCZOS), args.max_dimensions, args.min_dimensions)
                t = test_transform(image=np.array(pad(img).convert('RGB')))['image'][:1].unsqueeze(0)
                w = image_resizer(t.to(args.device)).argmax(-1).item()*32
                if (w/img.size[0] == 1):
                    break
                r *= w/img.size[0]
    else:
        img = np.array(pad(img).convert('RGB'))
        t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), args.max_seq_len,
                               eos_token=args.eos_token, context=encoded.detach(), temperature=args.get('temperature', .25))
        pred = post_process(token2str(dec, tokenizer)[0])
    # clipboard.copy(pred)
    return pred


def output_prediction(pred, args):
    print("Формула:")
    print(pred, '\n')

    if args.katex:
        # render using katex
        import webbrowser
        from urllib.parse import quote
        url = 'https://katex.org/?data=' + \
            quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
    "strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
        webbrowser.open(url)


if __name__ == "__main__":
    print("Использовать ли KaTeX для отображения результатов? (1 - да, 2 - нет)")
    katex = input().strip() == "1"

    latexocr_path = os.path.dirname(sys.argv[0])
    if latexocr_path != '':
        sys.path.insert(0, latexocr_path)
        os.chdir(latexocr_path)

    args, *objs = initialize(None, katex)
    while True:
        print('Введите путь к фото с формулой:')
        instructions = input()
        possible_file = instructions.strip()
        # ins = possible_file.lower()

        try:
            if os.path.isfile(possible_file):
                args.file = possible_file
            else:
                print("Неверный путь до файла, пожалуйста, введите другой путь")
                continue

            if args.file:
                img = Image.open(args.file)
            pred = call_model(args, *objs, img=img)
            output_prediction(pred, args)
        except KeyboardInterrupt:
            pass
        args.file = None
