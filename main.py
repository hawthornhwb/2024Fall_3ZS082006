from argparse import ArgumentParser


def trainer():
    pass


def test():
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_phase', action='store_true')

    args = parser.parse_args()
    if args.testphase:
        test()
    else:
        trainer()