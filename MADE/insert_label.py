import numpy as np
import os
import sys

def main(white_type, black_type, feat_type):

    print('insert label,', white_type, black_type, feat_type)

    root_dir = os.path.join('../data/', 'white_' + white_type + '_black_' + black_type, 'source')
    feat_dir = os.path.join(root_dir, 'feat')

    wp = np.load(os.path.join(feat_dir, 'wp_%s.npy'%(feat_type)))
    bp = np.load(os.path.join(feat_dir, 'bp_%s.npy'%(feat_type)))
    try:
        wn = np.load(os.path.join(feat_dir, 'wn_%s.npy'%(feat_type)))
        bn = np.load(os.path.join(feat_dir, 'bn_%s.npy'%(feat_type)))
    except:
        w_y = np.zeros(wp.shape[0])
        b_y = np.ones(bp.shape[0])
        w = np.concatenate((wp, w_y[:, None]), axis=1)
        b = np.concatenate((bp, b_y[:, None]), axis=1)
    else:
        w = np.concatenate([wp, bn], axis=0)
        w_y = np.concatenate(
            [
                np.zeros(wp.shape[0]),
                np.ones(bn.shape[0]),
            ], axis=0
        )
        w = np.concatenate((w, w_y[:, None]), axis=1)

        b = np.concatenate([wn, bp], axis=0)
        b_y = np.concatenate(
            [
                np.zeros(wn.shape[0]),
                np.ones(bp.shape[0]),
            ], axis=0
        )
        b = np.concatenate((b, b_y[:, None]), axis=1)

    np.random.shuffle(w)
    np.random.shuffle(b)

    np.save(os.path.join(feat_dir, 'w_%s.npy'%(feat_type)), w)
    np.save(os.path.join(feat_dir, 'b_%s.npy'%(feat_type)), b)

if __name__ == '__main__':
    _, white_type, black_type, feat_type = sys.argv 
    main(white_type, black_type, feat_type)