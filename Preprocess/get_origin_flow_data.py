import os
import sys
import numpy as np

def get_feat(file):
    try:
        fp = open(file, 'r')
    except:
        return None
    flows = []
    for i, line in enumerate(fp):
        line_s = line.strip().split(';')
        sq = line_s[0].split(',')
        feat = []
        for i in range(50):
            if i >= len(sq):
                feat.append(0)
            else:
                length = abs(int(sq[i]) - (0 if i == 0 else int(sq[i - 1])))
                if length >= 2000:
                    feat.append(1999)
                else:
                    feat.append(length)
        flows.append(feat)
    return np.array(flows, dtype=int)

def main(sequence_data_path, save_dir, data_type):

    data = get_feat(sequence_data_path)
    np.save(os.path.join(save_dir, data_type), data)

if __name__ == '__main__': 
    _, sequence_data_path, save_dir, data_type = sys.argv
    main(sequence_data_path, save_dir, data_type)
