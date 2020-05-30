import numpy as np;
class Instance:
    def __init__(self, N, M, K, Nmax, Mmax, Wmulti):
        self.N = N
        self.M = M
        self.K = K
        self.Nmax = Nmax
        self. Mmax = Mmax
        self.Wmulti = Wmulti

    def saveInstance(self, filename):
        N = self.N
        M = self.M
        K = self.K
        Nmax = self.Nmax
        Mmax = self.Mmax
        Wmulti = self.Wmulti

        f=open(filename, "w")
        f.write("%d %d %d %d %d \n" %(N, M, K, Nmax, Mmax))
        for W in Wmulti:
            np.savetxt(f, W) #np.savetxt permet d'écrire des tableaux
                              #1D ou 2D dans un fichier
            f.write("\n")
        f.close()

def LoadInstance(filename):
    f=open(filename,"r")
    lines = f.readlines()
    values = [int(x) for x in lines[0].split()]
    N, M, K, Nmax, Mmax = values[0], values[1], values[2], values[3], values[4]
    Wmulti=[]
    W = []
    for l in lines[1:]:
        if(l.strip()): #Vérifie que la ligne n'est pas vide
            W.append([float(x) for x in l.split()])
        else:
            Wmulti.append(W)
            W = []
    I = Instance(N, M, K, Nmax, Mmax, Wmulti)
    return I
