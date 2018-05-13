DATA='data/yeast.dat'
DATA_STARTS_FROM=121

def read_file(filename):
    with open(filename) as f:
        return [ l.strip() for l in f.readlines()[DATA_STARTS_FROM:] ]

def struc_data(filename=DATA):
    lines = read_file(filename)
    data = []
    for line in lines:
        slices = line.split(',')
        data.append((
                [ float(i) for i in slices[:103] ],
                [ int(i) for   i in slices[103:] ]
                ))
    return data


if __name__ == '__main__':
    data = struc_data(DATA)
    print(data[:2])
