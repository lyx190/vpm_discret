from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class Partial_iLIDS(object):
    
    def __init__(self, root):

        self.images_dir = osp.join(root)
        self.query_path = 'Probe'
        self.gallery_path = 'Gallery'
        self.query, self.gallery = [], []
        self.num_query_ids, self.num_gallery_ids = 0, 0
        self.load()

    def preprocess(self, path, relabel=True, is_probe=True):
        all_pids = {}
        ret = []
        img_paths = glob(osp.join(self.images_dir, path, '*.jpg'))

        for img_path in img_paths:
            pid = int(osp.basename(img_path).split('.')[0])
            if is_probe:
                cam = 0
            else:
                cam = 1
            if pid == -1: continue # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            ret.append((img_path, pid, cam))

        return ret, int(len(all_pids))
    
    def load(self):
        self.query, self.num_query_ids = self.preprocess(self.query_path, False, True)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))