# Saving files to disk

import pickle
def save_struct(obj, fname):
    pickling_on = open(fname,"wb")
    pickle.dump(obj, pickling_on)
    pickling_on.close()

def load_struct(fname):
    pickle_off = open(fname,"rb")
    return pickle.load(pickle_off)
