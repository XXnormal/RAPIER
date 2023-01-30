import os 
import sys 
sys.path.append('..')
import MADE
import CoTe
import AE

def generate(feat_dir, model_dir, made_dir, index, cuda):
    TRAIN_w = 'w_corrected'
    TRAIN_b = 'b_corrected'
    TRAIN = 'corrected'
    
    MADE.train.main(feat_dir, model_dir, TRAIN_w, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_b, cuda, '-30')
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_w, TRAIN_w, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_w, TRAIN_b, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_b, TRAIN_b, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_b, TRAIN_w, cuda)

    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, '1', index, cuda)

def generate_cpus(data_type, indices, cuda):
    for index in indices:
        generate(data_type, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda):
    
    AE.train.main(data_dir, model_dir, 'new', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'w', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'b', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)

    TRAIN = 'w_all'
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
    MADE.get_clean_epochs.main(feat_dir, made_dir, '1', TRAIN)
    MADE.final_predict.main(feat_dir)
    
    generate_cpus(feat_dir, model_dir, list(range(5)), cuda)
    
    TRAIN = 'corrected'
    CoTe.main_GAN_ensemble.main(feat_dir, model_dir, result_dir, TRAIN, cuda)
    
if __name__ == '__main__':
    data_dir = '../data/data'
    feat_dir = '../data/feat'
    model_dir= '../data/model'
    made_dir = '../data/made'
    result_dir='../data/result'
    cuda = '0'
    main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda)