import os
from multiprocessing import Pool as ThreadPool
from functools import partial
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Download APPs.')
    parser.add_argument('-f', '--file', help='The path of a txt contains some sha256.', type=str, required=True)
    parser.add_argument('-o', '--out', help='The path of output.', type=str, required=True)
    args = parser.parse_args()
    return args

def download(APP_Name, out, existing_files):
    if APP_Name in existing_files:
        return 0
    else:
        os.environ['APP_Name'] = str(APP_Name)
        os.system('curl -O --remote-header-name -G -d apikey=baa9c39f6ab08a9a35e64268685a139847b866bb9102a4a138177d98a717e461 -d sha256=$APP_Name https://androzoo.uni.lu/api/download')

        APK_name = APP_Name.upper() + '.apk'
        dir_ = os.getcwd()
        APK_path = dir_ + '/' + APK_name
        if os.path.isfile(APK_path):
            shutil.move(APK_path, out)

def main():
    args = parse_args()
    
    sha256 = []
    with open(args.file, 'r') as f:
        for line in f.readlines():
            sha256.append(line.strip().upper())
    while True:        
        existing_files = os.listdir(args.out)
        existing_files = [i.replace('.apk', '') for i in existing_files]

        pool = ThreadPool()
        pool.map(partial(download, out=args.out, existing_files=existing_files), sha256)
        if len(existing_files) == len(sha256):
            break

if __name__ == '__main__':
    main()

