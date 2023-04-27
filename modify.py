
import os

def main():
    pass
def make_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def running():
    move_dir = os.listdir(os.getcwd())
    for map in move_dir:
        if map == 'modify.py':
            continue
        new_lsit = []
        with open(map, 'r') as f:
            sample = f.read()
            for i in range(len(sample)):
                new_lsit.append(sample[i])
            new_lsit[111] = ''
            # 7*5 사이즈일시 121로 변경
        result = ""
        for s in new_lsit:
            result += s
        with open(map, 'w') as f:
            f.write(result)
        f.close()
def list_sorting():
    pass
if __name__ == '__main__':
    running()

