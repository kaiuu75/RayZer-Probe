import h5py

def load_nyu_depth_v2():
    path = 'nyu_depth_v2.h5y'
    h5py.File(path, 'r')