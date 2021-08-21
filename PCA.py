import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

d0=pd.read_csv('mnist_train.csv')
d0
a=d0.iloc[:,1:]. values
b=d0.iloc[:,0]. values
a.shape
l=d0['label']
d=d0.drop("label",axis=1)

print(l.shape)
print(d.shape)
def plot_digit(a,b,idx):
    img= a[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('true label: %d' %b[idx])
    plt.show()
    
plot_digit(a,b,3)

labels = l.head(15000)
data = d.head(15000)

print('The shape of the sample data is :',data.shape)

standardized_data=StandardScaler().fit_transform(data)
print(standardized_data.shape)

sample_data=standardized_data

cov_matrix= np.matmul(sample_data.T,sample_data)

print("The shape of covariance matrix is: ",cov_matrix.shape)

# eigenvalues in the end have the highest values

eig_values,eig_vectors = eigh(cov_matrix, eigvals=(782,783))
eig_vectors=eig_vectors.T
print("eigen eig_vectors",eig_vectors.shape)

new_coordinates=np.matmul(eig_vectors, sample_data.T)
print('new data points shape',new_coordinates.shape)

new_coordinates=np.vstack((labels,new_coordinates)).T
df=pd.DataFrame(data=new_coordinates, columns=('labels','first_principle_comp','second_principle_comp'))
df

sn.FacetGrid(df, hue='labels',height=6).map(plt.scatter,'first_principle_comp','second_principle_comp').add_legend()
plt.show()
