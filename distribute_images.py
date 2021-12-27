import shutil
import random
from os import listdir
from os.path import isfile, join

def main():
    who = '<Person name>'
    source = 'E:/<path to images of person>/' + who
    train = 'E:/faces/train/' + who
    valid = 'E:/faces/valid/' + who
    test = 'E:/faces/test/' + who

    files = [f for f in listdir(source) if isfile(join(source, f))]
    random.shuffle(files)

    for idx, file in enumerate(files):
        new_file = str(idx).zfill(4) + '.jpg'
        if idx < 2000:
            shutil.copyfile(join(source, file), join(train, new_file))
        elif idx < 2400:
            shutil.copyfile(join(source, file), join(valid, new_file))
        elif 2400 <= idx < 2460:
            shutil.copyfile(join(source, file), join(test, new_file))

if __name__ == "__main__":
    main()