import dotenv
import os


def load_file(path):

    if not os.path.isfile(path):
        raise FileNotFoundError()

    return dotenv.load_dotenv(path)
