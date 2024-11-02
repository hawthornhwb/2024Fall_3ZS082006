from argparse import ArgumentParser
import os
import shutil


def read_csv_labels(fname):
    """
    读取fname来给标签字典返回一个文件名
    example:
    input: '000bec180eb18c7604dcecc8fe0dba07', 'boston_bull'/n '001513dfcb2ffafc82cccf4d8bbaba97', 'dingo'
    output: {'000bec180eb18c7604dcecc8fe0dba07': 'boston_bull', '001513dfcb2ffafc82cccf4d8bbaba97': 'dingo'}
    """
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]

    return dict((name, label) for name, label in tokens)


def movefile(filename, target_dir):
    """将文件移动到目标目录"""

    os.makedirs(target_dir, exist_ok=True)
    shutil.move(filename, target_dir)


def train_valid_split(data_dir, labels, valid_ratio):
    """
    从原始训练集中拆分训练集与验证集
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    import collections
    n = collections.Counter(labels.values()).most_common()[-1][1]
    import math
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        if label not in label_count or label_count[label] < n_valid_per_label:
            movefile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            movefile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))


if __name__ == '__main__':
    # data_dir = 'kaggle_data_tiny'
    parser = ArgumentParser()
    parser.add_argument('-d','--data_dir', type=str, default="dog_breed_identification")
    parser.add_argument('-v', '--valid_ratio', type=float, default=0.2)
    args = parser.parse_args()
    labels = read_csv_labels(os.path.join(args.data_dir, 'labels.csv'))
    train_valid_split(args.data_dir, labels, args.valid_ratio)