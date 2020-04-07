import sys, os
import subprocess as sp
from multiprocessing import Pool


threads = 40
prefix = "preproc/tenten/"

def convert7z(name):
    with sp.Popen(["7zcat", name], stdout=sp.PIPE) as proc:
        with open(prefix + name.split("/")[-1] + ".txt", "w") as f:
            nospace = True
            for line in proc.stdout:        
                if not line:
                    continue
                line = line.decode()[:-1]
                if line[:2] == "<d":
                    f.write("\n")
                    nospace = True
                    continue
                if line == "<g/>":
                    nospace = True
                    continue
                if line == "</s>":
                    f.write("\n")
                    nospace = True
                    continue
                if line[0] == "<":
                    continue 
                if nospace:
                    f.write(line)
                    nospace = False
                else:
                    f.write(" "+line)

    print("Done", name)

with Pool(threads) as pool:
    pool.map(convert7z, sys.argv[1:])
