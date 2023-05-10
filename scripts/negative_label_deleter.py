import glob
import time
import multiprocessing
from tqdm import tqdm


def multip(i):
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    new_lines = []
    for j in lines:
        j = j.replace('\n', ' ')
        data = j.split(' ')
        d1 = float(data[1])
        d2 = float(data[2])
        d3 = float(data[3])
        d4 = float(data[4])
        if (d1 < 0.0 or d1 > 1.0):
            continue
        if (d2 < 0.0 or d2 > 1.0):
            continue
        if (d3 < 0.0 or d3 > 1.0):
            continue
        if (d4 < 0.0 or d4 > 1.0):
            continue
        new_lines.append(
            f'{data[0]} {str(d1)} {str(d2)} {str(d3)} {str(d4)}\n')
    f = open(i, 'w')
    f.writelines(new_lines)
    f.close()


start = time.time()
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    data = glob.glob('*.txt')
    num_files = len(data)
    with tqdm(total=num_files) as pbar:
        for _ in pool.imap_unordered(multip, data):
            pbar.update()
    pool.close()
    pool.join()
    print('Labels repaired in ' +
          str(time.time() - start) + ' seconds.')
