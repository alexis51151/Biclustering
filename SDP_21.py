# SDP program to assign objects to biclusters

import mosek
from   mosek.fusion import * # Simplier to use than the Optimizer API, with minimal performance loss compared to it
import Instance
import numpy as np # For the tests
import Kmeans
from math import *
from random import sample
import random


'''
Arguments for main function :
    - K : nb of biclusters
    - N : nb of objects
    - Nmax : max nb of objects in a cluster
    - W^k in W : . matrix of profits for cluster k (in position (i,j), you got the sum of the similarities for the objects i and j)
            . shape : N*N
Result:
    - X^k in X : Matrix of objects (in position (i,j), you got x_i^k * x_j^k)

'''

#Global variables :

# K = 2
# N = 3
# Nmax = N-1
# W = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2, 1], [0, 0, 0], [1, 2, 1]]]


I=Instance.LoadInstance("save3.txt")
K= I.K
N = I.N
Nmax = I.Nmax
W = I.Wmulti
M = I.M
Mmax = I.Mmax


def initProblem(N,K,Nmax): #Cette version n'est pas aléatoire uniforme

    if(K<ceil(N/Nmax)):
        raise ValueError('Trop peu de clusters pour le Nmax donné')
        return

    if(K>N):
        raise ValueError('Trop clusters pour le N donné')
        return

    tab = np.arange(0,K).tolist()
    count = np.zeros(K)
    X=np.zeros((N,K))

    for i in range(N):
        k=sample(tab,1)[0]
        X[i, k] = 1
        count[k]+=1
        if count[k]>=Nmax:
            tab.remove(k)
    return X


def assignSimilarities(X, N, K, M, Mmax, Wmulti):
    Y=np.zeros((M,K))
    for k in range(K):
        resultats = np.zeros(M)
        for m in range(M):
            res=0
            for i in range(N):
                for j in range(i+1,N): #i+1 pour ne pas compter deux fois ceux déjà pris en compte
                    res+=X[i,k]*X[j,k]*Wmulti[m][i][j]
            resultats[m]=res

        ind = np.argpartition(resultats, -Mmax)[-Mmax:] #Retourne les indices des Mmax plus grandes valeurs a vérifier
        Y[ind,k]=1
    return Y

def convertXtabToX(Xtab, N, K):
    i=0
    X = np.zeros((N,K))
    for tab in Xtab:
        X[:,i]=tab.diagonal()
        i+=1
    return X


def bicluster_SDP(X,K,N,M, Mmax, Nmax, W): # SDP resolution for a given bicluster
        
    
    # We only use dense matrix for now
    with Model("SDP") as SDP:
        # Constraint 22 and initialization
        Z = SDP.variable("Z", Domain.inPSDCone(N+1,K))
        for k in range(K):
            SDP.constraint(Z.index(k,0,0), Domain.equalsTo(1))
        e = Matrix.eye(N) # Identity matrix
        
        
        # Constraints (20,21,22)
        
        # Constraint n°20 (false in the article)
        somme = 0
        for k in range(K):
            Xk = Z.slice([k,1,1],[k+1,N+1,N+1]).reshape([N,N])
            somme = Expr.add(somme, Xk.diag())
        SDP.constraint(somme, Domain.equalsTo(np.ones(N)))

        # Constraint n°21
        for k in range(K):
            Xk = Z.slice([k,1,1],[k+1,N+1,N+1]).reshape([N,N])
            SDP.constraint(Expr.dot(e,Xk), Domain.lessThan(Nmax))
        
        # Constraint n°23
        
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    # Xkij = Z.slice([k,1+i,1+j],[k+1,2+i,2+j]).reshape(1)
                    Xkij = Z.index(k,1+i,1+j)
                    SDP.constraint(Xkij,Domain.lessThan(1))
                    SDP.constraint(Xkij,Domain.greaterThan(0))

        
        # Objective
        sum = dot_result_k(Z,W,N,0) # Sum for the objective function
        for k in range(1,K):
            dot_result = dot_result_k(Z,W,N,k)
            Expr.add(sum,dot_result)
            
        SDP.objective(ObjectiveSense.Maximize, sum)
        
        # Resolution
        SDP.solve()
        Xtab = []
        for k in range(K): # Print the clusters matrixes
            Xk = Z.slice([k,1,1],[k+1,N+1,N+1]).reshape([N,N])
            A = np.reshape(Xk.level(), [N,N])
            Xtab.append(A)
        return Xtab
    

def extract_xi(X):
    xi = []
    print("Vecteurs xi : ")
    for k in range(len(X)):
        values, vectors= np.linalg.eig(X[k])
        ind=np.argmax(values)
        x=vectors[ind]
        xi.append(np.sqrt(values[ind])*x)
    for elt in xi:
         print(elt)
    return(xi)

def extract_xi_diag(X):
    xi = []
    #print("Vecteurs xi obtenus par racine de la diagonale: ")
    for k in range(len(X)): 
        line = np.zeros(N)
        for i in range(N): # processing the diag
            line[i] = np.sqrt(X[k][i][i])
        xi.append(line)
    #print(xi)
    return(xi)

def construct_Wks(X,N,K,M,Mmax,W):
    Wtab = [0] * K
    Y = assignSimilarities(X,N,K,M,Mmax,W)
    # Calculating the elements of Wk
    for k in range(K):
        Wk = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                for l in range(M):
                    Wk[i][j] += Y[l][k]*W[l][i][j]
        Wtab[k] = Wk
    return Wtab,Y

def dot_result_k(Z,W,N,k): # k from 0 to K-1
    Xk = Z.slice([k,1,1],[k+1,N+1,N+1]).reshape([N,N])
    Wk = W[k]
    return(Expr.dot(Xk,Wk))


def map_to_X(KMmap):
    X = np.zeros((N,K))
    for i in range(N):
        pos = KMmap[i]
        X[i][pos] = 1
    return X

def main():
    
    X = initProblem(N, K, Nmax)
    for i in range(10):
        W2,Y = construct_Wks(X,N,K,M,Mmax,W)
    
        Xtab = bicluster_SDP(X,K,N,M,Mmax,Nmax,W2)
        
        xi = extract_xi_diag(Xtab)
        # Converting for k-means
        kmeansVars = [np.zeros(K) for i in range(N)]
        for i in range(K): # For each cluster
            for j in range(N): # For each coord
                kmeansVars[j][i] = xi[i][j]
        # Using k-means constrained
        KM = Kmeans.KmeansConstrained(kmeansVars, Nmax, N, K, 100)
        test = [np.array([-5,0]), np.array([-3,0]), np.array([-4,0]), np.array([2,0]), np.array([3,0])]
        #KM = Kmeans.KmeansConstrained(test, 3, 5, 2, 50)
        KM.initialization()
        KM.assignment(0)
        X = map_to_X(KM.map)
    print(Y)
    print(KM.map)


    