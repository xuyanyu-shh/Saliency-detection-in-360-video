import numpy as np
import torch as th
import torch.utils.data as data
from PIL import Image
import os
import pickle
from scipy import signal
from sconv.functional.sconv import spherical_conv
from tqdm import tqdm
import numbers
import cv2
from functools import lru_cache
from random import Random


class VRSaliency(data.Dataset):
    def __init__(self, root, frame_h, frame_w, frame_interval=1, video_chosen=None, video_exclude=None, transform=None,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = set()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.add(vid)
        assert set(self.vinfo.keys()) == vset
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_chosen, set):
            vset = vset.intersection(video_chosen)
        elif isinstance(video_chosen, numbers.Integral):
            vset = set(rnd.sample(vset, k=video_chosen))
        if video_exclude:
            vset = vset - set(video_exclude)
        print('{} videos chosen.'.format(len(vset)))

        self.data = []
        self.target = []
        for vid in tqdm(vset, desc='video'):
            obj_path = os.path.join(root, vid)
            fcnt = 0
            for frame in tqdm(os.listdir(obj_path), desc='frame({})'.format(vid)):
                if frame.endswith('.jpg'):
                    fid = frame[:-4]
                    if fid not in self.vinfo[vid].keys():
                        print('warn: video {}, frame {} have no gt, abandoned.')
                        continue
                    fcnt += 1
                    if fcnt >= frame_interval:
                        self.data.append(os.path.join(obj_path, frame))
                        self.target.append(self.vinfo[vid][fid])
                        fcnt = 0

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)
        target = self._get_salency_map(item)

        return img, target

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if self.cache_gt and os.path.isfile(cfile):
            target_map = th.from_numpy(np.load(cfile)).float()
            assert target_map.size() == (1, self.frame_h, self.frame_w)
            return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideo(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_salency_map(-1)
        else:
            last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

        target = self._get_salency_map(item)

        if self.train:
            return img, last, target
        else:
            return img, self.data[item], last, target

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideoS2CNN(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_salency_map(-1)
        else:
            last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

        target = self._get_salency_map(item)

        if self.train:
            return img, last, target
        else:
            return img, self.data[item], last, target

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            pass
            # if self.cache_gt and os.path.isfile(cfile):
            #     target_map = np.load(cfile)
            #     if not target_map.size() == (1, self.frame_h, self.frame_w):
            #         target_map = cv2.resize(target_map[0, :, :], (self.frame_w, self.frame_h)).reshape(1, self.frame_h, self.frame_w)
            #     return th.from_numpy(target_map).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class ICMEDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        data_dir = os.path.join(root, 'train' if train else 'eval')
        self.transform = transform
        self.train = train

        self.img = []
        self.target = []
        for file in tqdm(os.listdir(data_dir), desc='scanning dir'):
            if file.endswith('.bin'):
                self.target.append(os.path.join(data_dir, file))
                self.img.append(os.path.join(data_dir, 'P' + file[3:-4] + '.jpg'))

    def __getitem__(self, item):
        img = Image.open(open(self.img[item], 'rb'))
        # print(self.img[item], flush=True)
        img_shape = np.array(img).shape[:2]
        target = np.fromfile(self.target[item], dtype=np.float32).reshape(*img_shape)
        target = cv2.resize(target, (256, 128)).reshape(1, 128, 256)

        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, th.from_numpy(target).float()
        else:
            _, filename = os.path.split(self.target[item])
            return img, filename[:-4], th.from_numpy(target).float()

    def __len__(self):
        return len(self.img)


class VRVideoImproved(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643, tmp_root='./'):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.tmp_root = tmp_root

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_salency_map(-1)
            last_pred = last
        else:
            last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])
            if os.path.isfile(os.path.join(self.tmp_root, vid, ('%04d' % (int(fid) - self.frame_interval)) + '.bin')):
                # print('use last pred map.')
                last_pred = np.fromfile(
                    os.path.join(self.tmp_root, vid, ('%04d' % (int(fid) - self.frame_interval)) + '.bin'),
                    dtype=np.float32).reshape(128, 256)
                last_pred = th.from_numpy(cv2.resize(last_pred, (256, 128)).reshape(1, 128, 256)).float()
            else:
                last_pred = last

        target = self._get_salency_map(item)

        return img, last, last_pred, target, vid, fid

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideoImprovedJoint(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643, tmp_root='./'):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.tmp_root = tmp_root

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        # vid, fid = self.i2v[item]
        # if int(fid) - self.frame_interval <= 0:
        #     last = self._get_salency_map(-1)
        #     last_pred = last
        # else:
        #     last = self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])
        #     if os.path.isfile(os.path.join(self.tmp_root, vid, ('%04d' % (int(fid) - self.frame_interval)) + '.bin')):
        #         # print('use last pred map.')
        #         last_pred = np.fromfile(
        #             os.path.join(self.tmp_root, vid, ('%04d' % (int(fid) - self.frame_interval)) + '.bin'),
        #             dtype=np.float32).reshape(128, 256)
        #         last_pred = th.from_numpy(cv2.resize(last_pred, (256, 128)).reshape(1, 128, 256)).float()
        #     else:
        #         last_pred = last

        target = self._get_salency_map(item)

        return img, target / target.max()

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideoRotTest(data.Dataset):
    def __init__(self, root, frame_h, frame_w, frame_interval=5, transform=None):
        self.root = root
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w

        self.gaussian_sigma = np.pi / 20
        self.kernel_rad = np.pi / 7
        self.kernel_size = (30, 60)
        self.cache_gt = False

        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in os.listdir(root):
            for frame in os.listdir(os.path.join(root, vid)):
                if frame.endswith('.jpg'):
                    fid = frame[:-4]
                    self.i2v[len(self.data)] = (vid, fid)
                    self.v2i[(vid, fid)] = len(self.data)
                    self.data.append(os.path.join(root, vid, frame))
                    self.target.append(os.path.join(root, vid, fid + '.bin'))

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        h, w, _ = np.array(img).shape
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_salency_map(-1)
        else:
            last = np.fromfile(os.path.join(self.root, vid, ('%04d' % (int(fid) - self.frame_interval)) + '.bin'),
                               dtype=np.float32).reshape(h, w)
            last = th.from_numpy(cv2.resize(last, (self.frame_w, self.frame_h)).reshape(1, self.frame_h, self.frame_w)).float()

        target = np.fromfile(self.target[item], dtype=np.float32).reshape(h, w)
        target = th.from_numpy(cv2.resize(target, (self.frame_w, self.frame_h)).reshape(1, self.frame_h, self.frame_w)).float()

        return img, last, target, vid, fid

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=None)
    def _get_salency_map(self, item, use_cuda=False):
        assert item == -1
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideoMultiFrame(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True,
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

    def __getitem__(self, item):
        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)

        last = []

        vid, fid = self.i2v[item]
        for step in range(1, 6):
            if int(fid) - self.frame_interval * step <= 0:
                last.append(self._get_salency_map(-1))
            else:
                last.append(self._get_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval * step))]))

        target = self._get_salency_map(item)
        last = th.cat(last, dim=0)

        if self.train:
            return img, last, target
        else:
            return img, self.data[item], last, target

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRRotatedTest(data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        self.img = []
        self.target = []
        self.i2v = {}
        for vid in tqdm(os.listdir(root), desc='video'):
            for fid in tqdm(os.listdir(os.path.join(root, vid)), desc='frame'):
                file = os.path.join(root, vid, fid)
                if file.endswith('.jpg'):
                    self.i2v[len(self.img)] = (vid, fid[:-4])
                    self.target.append(file[:-4] + '.bin')
                    self.img.append(file)

    def __getitem__(self, item):
        vid, fid = self.i2v[item]
        img = Image.open(open(self.img[item], 'rb'))
        img_shape = np.array(img).shape[:2]
        target = np.fromfile(self.target[item], dtype=np.float32).reshape(*img_shape)
        target = cv2.resize(target, (256, 128)).reshape(1, 128, 256)

        if self.transform:
            img = self.transform(img)

        return img, th.from_numpy(target).float(), vid, fid

    def __len__(self):
        return len(self.img)


if __name__ == '__main__':

    def gen_gaussian_kernel(sigma_idx=8, kernel_size=(15, 30)):
        gauss1d = signal.gaussian(2 * kernel_size[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel_size[1]))

        return gauss2d[-kernel_size[0]:, :]


    import matplotlib.pyplot as plt
    # h, w = 30, 15
    # kernel = gen_gaussian_kernel(sigma_idx=4)
    # for test_h in range(h):
    #     img = np.zeros((h, w))
    #     img[test_h, int(w/2)] = 1
    #     target_map = spherical_conv(
    #         th.from_numpy(
    #             img.reshape(1, 1, *img.shape)
    #         ),
    #         th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
    #         kernel_rad=np.pi/5
    #     ).view(h, w).data.numpy()
    #     # print(test_h)
    #     cv2.imshow('res', target_map*255)
    #     cv2.waitKey(500)


    import matplotlib.pyplot as plt
    # dataset = VRSaliency('/home/ziheng/dataset-beta-v2.0-jpg', 150, 300, cache_gt=False).cache_map()
    # img, map = dataset[5]
    #
    # fix, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img)
    # ax2.imshow(map.numpy().reshape(150, 300))
    # plt.show()

    dataset = ICMEDataset('/home/ziheng/2018-ECCV/ICME')
    img, map = dataset[11]
    fix, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(map)
    plt.show()

