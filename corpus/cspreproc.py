import sys
import re
import random

from os import listdir
from os.path import isfile, join, basename
from multiprocessing import Pool

THREADS = 6 
MAX_LENGTH = 500000000

def preproc(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = re.sub("[^0-9a-zóěščřžýáíďéťňůú\\.,\\!\\?% ]", "", sentence)
    sentence = re.sub("[0-9]+", "#", sentence)
    #sentence = re.sub(r"([^\. #])([\.,?!])", r"\1 .", sentence)
    return sentence

def preproc_file(path):
    print("starting", path)
    length = 0
    with open(path) as infile:
        with open(join(outdir, basename(path)), "w") as outfile:
            text = infile.read() 
            docs = text.split("\n\n") 
            random.shuffle(docs) 
            print("loaded  ", path, len(docs))
            for doc in docs:
                for line in doc.split("\n"):
                    #print(len(line))
                    line = preproc(line)
                    outfile.write(line)
                    outfile.write("\n")
                    length += len(line)
                if length > MAX_LENGTH:
                    break
                outfile.write("\n")
                
    print("finished", path)

if __name__ == "__main__":
    _, indir, outdir = sys.argv
    files = [join(indir, f) for f in listdir(indir) if isfile(join(indir, f))]
    with Pool(THREADS) as pool:
        pool.map(preproc_file, files)
    
    


 
