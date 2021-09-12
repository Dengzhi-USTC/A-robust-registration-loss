"""
evaluate the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
from se_math.se3 import exp
import model
import argparse
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('./../')
sys.path.append('./../../')
from pre_dataloader import generate_datasets_human

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # used GPU card no.


# visualize the point clouds
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    open3d.io.write_point_cloud("source_pre.ply", source_temp)
    source_temp.transform(transformation)
    open3d.io.write_point_cloud("source.ply", source_temp)
    open3d.io.write_point_cloud("target.ply", target_temp)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def options(argv=None):
    parser = argparse.ArgumentParser(description='Feature-metric registration')

    # required to check
    parser.add_argument('-data',
                        '--dataset-type',
                        default='7scene',
                        choices=['modelnet', '7scene'],
                        metavar='DATASET',
                        help='dataset type (default: modelnet)')
    parser.add_argument('-o',
                        '--outfile',
                        default='./result/result.csv',
                        type=str,
                        metavar='FILENAME',
                        help='output filename (.csv)')
    parser.add_argument('-p',
                        '--perturbations',
                        default='./data/pert_010.csv',
                        type=str,
                        metavar='PATH',
                        help='path to the perturbation file')  # run
    parser.add_argument('-l',
                        '--logfile',
                        default='./result/log_010.log',
                        type=str,
                        metavar='LOGNAME',
                        help='path to logfile (default: null (no logging))')
    parser.add_argument(
        '--pretrained',
        default='',
        type=str,
        metavar='PATH',
        help='path to trained model file (default: null (no-use))')

    # settings for performance adjust
    parser.add_argument('--max-iter',
                        default=5,
                        type=int,
                        metavar='N',
                        help='max-iter on IC algorithm. (default: 20)')
    parser.add_argument('--dim-k',
                        default=1024,
                        type=int,
                        metavar='K',
                        help='dim. of the feature vector (default: 1024)')
    # settings for on testing
    parser.add_argument('-j',
                        '--workers',
                        default=2,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        metavar='DEVICE',
                        help='use CUDA if available (default: cpu)')
    parser.add_argument(
        '-i',
        '--dataset-path',
        default='',
        type=str,
        metavar='PATH',
        help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('-c',
                        '--categoryfile',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to the categories to be tested'
                        )  # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('--mode',
                        default='test',
                        help='program mode. This code is for testing')

    args = parser.parse_args(argv)
    return args


def main(args):
    # dataset
    # testset = dataset.get_datasets(args)
    data_path = "/data1/dengzhi/src_tar_datasets2"
    data_path = "/data2/dengzhi/Airplane_datasets/src_tar_datasets2"

    data_path = "/data2/dengzhi/Airplane_datasets/src_tar_datasets2"

    # trainset = generate_datasets_real_test_data()
    train_datasets, test_datasets = generate_datasets_human()  # testing
    expname = 'Test'

    label = "real_data_cf_1"

    if not os.path.exists(expname):
        os.makedirs(expname)
    save_results = os.path.join(expname, label)
    if not os.path.exists(save_results):
        os.makedirs(save_results)

    test_writer = SummaryWriter(os.path.join(save_results, './logs'))
    # generate the writer, make a test!
    fmr = model.FMRTest(args)

    run(args,
        test_datasets,
        fmr,
        save_results=save_results,
        writer=test_writer)


def run(args, testset, action, save_results=None, writer=None):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
    decoder_dict = torch.load(args.pretrained)

    decoder_keys = [
        key for key in decoder_dict.keys() if key.startswith('decoder.')
    ]
    for key in decoder_keys:
        del decoder_dict[key]

    model.load_state_dict(decoder_dict)
    # print("load the model")
    model.to(args.device)
    model.eval()

    testloader = testset
    # testing
    LOGGER.debug('tests, begin')
    action.evaluate(model,
                    testloader,
                    args.device,
                    save_results=save_results,
                    writer=writer)
    LOGGER.debug('tests, end')


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())
