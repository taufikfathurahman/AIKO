import os

def execute_rename():
    BASE_DIR = '../dataset'
    list_dir = os.listdir(BASE_DIR)
    index = 0
    for dirname in list_dir:
        i = 0
        for filename in os.listdir(BASE_DIR + '/' + dirname):
            dst = dirname + "_" + str(i) + ".jpg"
            src = BASE_DIR + '/' + dirname + '/' + filename
            dst = BASE_DIR + '/' + dirname + '/' + dst
            try:
                os.rename(src, dst)
            except:
                print(dst, 'already exist')
            i += 1
        index += 1

if __name__=='__main__':
    execute_rename()