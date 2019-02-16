from torch import nn
import numpy as np
import torch as th
from data import VRVideo
import torchvision.transforms as tf
from torch.utils import data as tdata
from torch.optim import SGD
from torch.autograd import Variable
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
import visdom
import time

from spherical_unet import Final1
from sconv.module import SphericalConv, SphereMSE


def train(
        data,
        bs=28,
        lr=3e-1,
        epochs=100,
        clear_cache=False,
        plot_server='http://127.0.0.1',
        plot_port=9088,
        save_interval=100,
        resume=True,
        start_epoch=0,
        exp_name='final',
        test_mode=False
):

    viz = visdom.Visdom(server=plot_server, port=plot_port, env=exp_name)

    transform = tf.Compose([
        tf.Resize((128, 256)),
        tf.ToTensor()
    ])
    dataset = VRVideo(data, 128, 256, 80, frame_interval=5, cache_gt=True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    if clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    model = Final1()
    optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    pmodel = nn.DataParallel(model).cuda()
    criterion = SphereMSE(128, 256).float().cuda()
    if resume:
        ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

    log_file = open(exp_name +'.out', 'w+')
    for epoch in trange(start_epoch, epochs, desc='epoch'):
        tic = time.time()
        for i, (img_batch, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
            img_var = Variable(img_batch).cuda()
            last_var = Variable(last_batch * 10).cuda()
            t_var = Variable(target_batch * 10).cuda()
            data_time = time.time() - tic
            tic = time.time()

            out = pmodel(img_var, last_var)
            loss = criterion(out, t_var)
            fwd_time = time.time() - tic
            tic = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bkw_time = time.time() - tic

            msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
                epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0]
            )
            viz.images(target_batch.cpu().numpy() * 10, win='gt')
            viz.images(out.data.cpu().numpy(), win='out')
            viz.text(msg, win='log')
            # print(msg, file=log_file, flush=True)
            print(msg, flush=True)

            tic = time.time()

            if (i + 1) % save_interval == 0:
                state_dict = model.state_dict()
                ckpt = dict(epoch=epoch, iter=i, state_dict=state_dict)
                th.save(ckpt, 'ckpt-' + exp_name + '-latest.pth.tar')


if __name__ == '__main__':
    Fire(train)
