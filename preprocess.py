import os
import hydra
import librosa
import utils
from hydra.utils import to_absolute_path, get_original_cwd
from os.path import expanduser, exists, basename, join, dirname
from utils import read_filelist, write_filelist, find_all_files
from tqdm import tqdm

@hydra.main(version_base=None, config_path='config', config_name='default')
def preprocess(cfg):
    print(f"Hydra run dir: {os.getcwd()}")
    print(f"Original cwd : {get_original_cwd()}")

    # 입력 루트와 출력 파일 경로를 '원래 CWD' 기준 절대경로로 고정
    root = to_absolute_path(expanduser(cfg.preprocess.datasets.LibriSpeech.root))
    train_out = to_absolute_path(cfg.preprocess.view.train_filelist)
    val_out   = to_absolute_path(cfg.preprocess.view.val_filelist)
    test_out  = to_absolute_path(cfg.preprocess.view.test_filelist)

    # 부모 디렉토리 보장
    os.makedirs(dirname(train_out), exist_ok=True)
    os.makedirs(dirname(val_out), exist_ok=True)
    os.makedirs(dirname(test_out), exist_ok=True)

    trainfiles, valfiles, testfiles = [], [], []
    print(f'Root: {root}')

    for subset in cfg.preprocess.datasets.LibriSpeech.trainsets:
        files = find_all_files(join(root, subset), '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        for i in range(len(files)):
            files[i][1] = files[i][1].replace(root, '').lstrip('/')
        trainfiles.extend(files)

    print(f'Write train filelist to {train_out}')
    utils.write_filelist(trainfiles, train_out)

    for subset in cfg.preprocess.datasets.LibriSpeech.valsets:
        files = find_all_files(join(root, subset), '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        for i in range(len(files)):
            files[i][1] = files[i][1].replace(root, '').lstrip('/')
        valfiles.extend(files)
    print(f'Write val filelist to {val_out}')
    utils.write_filelist(valfiles, val_out)

    for subset in cfg.preprocess.datasets.LibriSpeech.testsets:
        files = find_all_files(join(root, subset), '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        for i in range(len(files)):
            files[i][1] = files[i][1].replace(root, '').lstrip('/')
        testfiles.extend(files)
    print(f'Write test filelist to {test_out}')
    utils.write_filelist(testfiles, test_out)

if __name__ == '__main__':
    preprocess()