import os
import numpy as np
import pdb, argparse
import sys, random
import copy, torch
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.train import train_X_to_C_to_y, train_X_to_C

def parse_arguments():
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('-exp', default=None, help='which experiment to run')
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-seed', default=1, type=int, help='Numpy and torch seed.')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
    parser.add_argument('-weighted_loss', default='',
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
    parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-uncertain_labels', action='store_true',
                        help='whether to use (normalized) attribute certainties as labels')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
                    
    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.exp == "joint":
        train_X_to_C_to_y(args)
    if args.exp == "XtoC":
        train_X_to_C(args)

# python3 experiments.py -exp joint -log_dir CUB/outputs/ -seed 1 -ckpt 1 -e 100 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 33 -end2end
# python3 experiments.py -exp joint -log_dir CUB/outputs/ -seed 2 -ckpt 1 -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20 -end2end

# python3 experiments.py -exp XtoC -seed 1 -ckpt 1 -log_dir CUB/outputs/conceptModel -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck