import os 
import sys 
sys.path.append('..')
import MADE
import Classifier
import AE

def generate(feat_dir, model_dir, made_dir, index, cuda):
    TRAIN_be = 'be_corrected'
    TRAIN_ma = 'ma_corrected'
    TRAIN = 'corrected'
    
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)

    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda):
    
    AE.train.main(data_dir, model_dir, cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)

    TRAIN = 'be'
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
    MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
    MADE.final_predict.main(feat_dir)
    
    generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    
    TRAIN = 'corrected'
    Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=5)
    
if __name__ == '__main__':
    data_dir = '../data/data'
    feat_dir = '../data/feat'
    model_dir= '../data/model'
    made_dir = '../data/made'
    result_dir='../data/result'
    cuda = 5
    main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda)