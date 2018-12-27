import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jpg_path',
        default='Resque/Labeled/NoLabeled',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--annotation_file',
        # default='Resque/Labeled/NoLabeled/annotation.txt',
        default='Annotation/20150801',
        type=str,
        help='Annotation file path [Annotation/20150810 | Annotation/20160710 | Annotation/20161111]')
    parser.add_argument(
        '--result_path',
        default='./Result/',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--n_classes',
        default=11,
        type=int,
    )
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--gpu', default=0, type=str)
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument(
        '--epochs',
        default=60,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--num_works', default=30, type=int, help='Number of works')
    parser.add_argument(
        '--threthold', default=0.5, type=float, help='threthold for N-ok-K')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--save_name',
        default="resquedog",
        type=str,
        help='Core name to save for: image, logs and other outputs')
    parser.add_argument(
        '--mulch_gpu',
        help='If true use mulch GPU.')
    parser.set_defaults(mulch_gpu=False)

    
    args = parser.parse_args()

    return args
