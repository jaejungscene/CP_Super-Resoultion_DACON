import argparse
from datetime import datetime
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_args_parser():
    parser = argparse.ArgumentParser(description='[DACON] YangGae hub, super resolution')
    parser.add_argument('--model', default='srcnn', type=str,
                    help='type of network') 
    parser.add_argument('--batchSize', default=12, type=int,
                    help='number of fold')
    parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')    
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=1e-4, type=float,
                    help='write learning rate')
    parser.add_argument('--imgSize', default=2048, type=int,
                    help="write input image size")
    parser.add_argument('--foldNum', default=0, type=int,
                    help='number of fold')
    parser.add_argument('--workersNum', default=3, type=int,
                    help='number of fold')
    parser.add_argument('--cuda', type=str, default='0', 
                    help='select used GPU')
                    
    parser.add_argument('--expname', default=result_folder_name, type=str,
                    help='name of experiment')
    parser.add_argument('--seed', default=41, type=int,
                    help='seed number')
    parser.add_argument('--wandb', type=int, default=0, 
                    help='choose activating wandb')

    return parser