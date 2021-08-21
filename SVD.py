#Sahil Aggarwal 
#Importing Libraries

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier
X, y = load_digits(return_X_y=True)

#Original
image = X[0]
image = image.reshape((8, 8))
plt.matshow(image, cmap = 'gray')
plt.show()
#Reduced
U, s, V = np.linalg.svd(image)
S = np.zeros((image.shape[0], image.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
n_component = 2
S = S[:, :n_component]
V = V[:n_component, :]
A = U.dot(S.dot(V))


plt.matshow(A, cmap = 'gray')
plt.show()


U.dot(S)

#compressing multiple images from a datset
def svd_on_dataset(comp):
    i=0
    plt.figure(figsize=(2, 2))
    while i<300:
        image = X[i]
        image = image.reshape((8, 8))
        plt.matshow(image, cmap = 'gray')
        
       
        U, s, V = np.linalg.svd(image)
        S = np.zeros((image.shape[0], image.shape[1]))
        S[:image.shape[0], :image.shape[0]] = np.diag(s)
        n_component = 2
        S = S[:, :n_component]
        V = V[:n_component, :]
        A = U.dot(S.dot(V))
        plt.matshow(A, cmap = 'gray')
        plt.show()
        i=i+1
        
       
svd_on_dataset(2)#
