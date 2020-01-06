import dotenv
import os


class Config():

    @property
    def seed(self):
        return int(os.getenv('SEED', 42))

    @property
    def k_folds(self):
        return int(os.getenv('K_FOLDS', 5))

    @property
    def pipline_fnt(self):
        return os.getenv('PIPELINE_FNT', 'build_sklearn_pipline')

    def __str__(self):
        return 'seed: %s, k_folds: %s, pipline_fnt: %s' % \
            (self.seed, self.k_folds, self.pipline_fnt)


DEFAULT = Config()


def set_global_seed(seed=None):

    import numpy as np
    import random as rd

    if not seed:
        seed = DEFAULT.seed

    np.random.seed(seed)
    rd.seed(seed)


def load_file(path):

    if not os.path.isfile(path):
        print('%s not found.' % path)
        print('Default config loaded.')
        print(DEFAULT)

    return dotenv.load_dotenv(path)
