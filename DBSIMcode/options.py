import argparse
from enum import Enum

class CoeffiType(Enum):
    RDC = 0
    PCC = 1

class Dataset(Enum):
    IMDB = 0
    TPCH = 1
    CENSUS = 2
    DMV = 3


parser = argparse.ArgumentParser()

parser.add_argument('--coeffi', type=float, default=0.05, help='threshold of correlation coefficient for columns splitting')
parser.add_argument('--cluster_per_col', type=int, default=13, help='the number of cluster per column')
parser.add_argument('--n_buckets', type=int, default=20, help='number of buckets for a histogram')
parser.add_argument('--muta_dist', type=int, default=200, help='distance for query generation')
# parser.add_argument('--eval', action='store_true', help='perform evaluation or not')
parser.add_argument('--eval', type=bool, default=False, help='perform evaluation or not')
parser.add_argument('--scale_factor', type=int, default=1, help='factor for scaling up')
parser.add_argument('--type_coeffi', choices=[x.name.upper() for x in CoeffiType], default='RDC', help='choose a coefficient between random dependency coefficient and pearson correlation coefficient')
parser.add_argument('--dataset', choices=[x.name.upper() for x in Dataset], default='CENSUS', help='choose a dataset')
# parser.add_argument('--verbose', action='store_true', help='increase output verbosity')
parser.add_argument('--verbose', type=bool, default=False, help='increase output verbosity')
parser.add_argument('--gen_dir', type=str, default='test',help='the name of the directory in which data generated will be stored')

parser.add_argument('--in_dir', type=str, default='../../samples/', help='the directory of input dataset')


parser.add_argument('--sche_file', type=str, default='./schema_census.txt', help='the schema file')



parser.add_argument('--query_file', type=str, default='../../queries/mscn_400.csv', help='csv file of queries')
parser.add_argument('--repeat_join_threshold', type=int, default=0, help='frequency threshold to record the exact join key')
parser.add_argument('--repeat_threshold', type=int, default=3, help='frequency threshold to record the exact value')
parser.add_argument('--privacy_budget', type=float, default=0.1, help='privacy budget for adding noise')


opt = parser.parse_args()