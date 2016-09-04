import tarfile
import numpy as np

tar = tarfile.open("test.tar.gz", "r:gz")
for member in tar.getmembers():
    f = tar.extractfile(member)
    content=f.read()
    print np.frombuffer(content)
        
tar.close()
