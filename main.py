#coding=utf-8

import argparse
import os
import time

from skipgram import build_model, train_op, train
from dataset import Dataset

def parse_args():
    # parse ht arguments
    parser = argparse.ArgumentParser(description='metapath2vec')
    parser.add_argument('--walks', type=str, required=True, help='text file that has a random walk in each line.')
    parser.add_argument('--types', type=str, required=True, help='text file that has node types.')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs.')
    # parser.add_argument('--batch', type=int, default=1, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--log', type=str, required=True, help='log directory.')
    parser.add_argument('--log-interval', type=int, default=-1, help='log intervals. -1 means per epoch')
    parser.add_argument('--max-keep-model', default=10, type=int, help='number of models to keep saving')
    parser.add_argument('--embedding-dim', default=100, type=int, help='embedding dimensions')
    parser.add_argument('--negative-samples', default=5, type=int, help='number of negative samples')
    parser.add_argument('--care-type', default=1, type=int,
                        help='care type or not. if 1, it cares (i.e. heterogeneous negative sampling). If 0, it does not care (i.e. normal negative sampling).')
    parser.add_argument('--window', default=5, type=int, help='context window size')

    return parser.parse_args()

def main(args):
    if os.path.isdir(args.log):
        print('%s already exist. Are you sure to override? I will wait for 5 seconds. Ctrl-C to abort.' % args.log)
        time.sleep(5)
        os.system('rm -rf %s/' % args.log)
    else:
        os.makedirs(args.log)
        print('make the log directory %s success!' % args.log)

    t0 = time.time()
    dataset = Dataset(random_walk_txt=args.walks, node_type_mapping_txt=args.types, window_size=args.window)
    center_node_placeholder, context_node_placeholder, negative_sample_placeholder, loss = build_model(BATCH_SIZE=1, VOCAB_SIZE=len(dataset.nodeid2index), EMBED_SIZE=args.embedding_dim, NUM_SAMPLED=args.negative_samples)
    optimizer = train_op(loss, LEARNING_RATE=args.lr)
    train(center_node_placeholder, context_node_placeholder, negative_sample_placeholder, loss, dataset, optimizer, NUM_EPOCHS=args.epochs, BATCH_SIZE=1, NUM_SAMPLED=args.negative_samples, care_type=args.care_type, LOG_DIRECTORY=args.log, LOG_INTERVAL=args.log_interval, MAX_KEEP_MODEL=args.max_keep_model)
    t1 = time.time()
    deltaT = t1 - t0
    deltaT = deltaT / 60.0
    print('Totally spend time is: %f min' % deltaT)

if __name__ == '__main__':
    args = parse_args()
    main(args)