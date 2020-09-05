from config.env import *


class data_dir:
    @staticmethod
    def make_raw_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'landing_zone', 'raw', *folders)

    @staticmethod
    def make_zips_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'landing_zone', 'zips', *folders)

    @staticmethod
    def make_processed_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'processed', *folders)

    @staticmethod
    def make_feature_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'processed', 'features', *folders)

    @staticmethod
    def make_interim_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'interim', *folders)

    @staticmethod
    def make_data_check_path(*folders):
        return os.path.join('.', DATA_FOLDER, 'data_checks', *folders)
