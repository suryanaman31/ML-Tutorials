#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import csv
path_to_Ar = 'D:\\Data_for_DL_and_ML\\Ar.txt'

n_atoms = 6
n_lines = 0
a_periodic = 12

raw_list = []
with open(path_to_Ar,'r') as Ar_data:
    reader = csv.reader(Ar_data, delimiter='\t')
    for row in reader:
        raw_list.append(row)
        n_lines+=1
n_snaps = int(n_lines/(n_atoms+2))

list_cleaned = []
for i in range(len(raw_list)):
    if raw_list[i][0]=='Ar':
        list_cleaned.append(raw_list[i])
        
chunks = lambda lst, atoms: [lst[i:i+atoms] for i in range(0, len(lst), atoms)]
NN_chunks = chunks(list_cleaned,n_atoms) #n_atoms is 6 for this case

#convert string to float values in the list (all entries except Ar)
for i in range(n_snaps):
    for j in range(n_atoms):
        for k in range(1,4):
            NN_chunks[i][j][k]=float(NN_chunks[i][j][k])
#remove = [list_chunks[i][j].pop(0) for j in range(n_atoms) for i in range(n_snaps)]

print(NN_chunks)

#Computation of Rij matrix
X_interatomic = np.zeros((n_snaps,n_atoms,n_atoms))
for n in range(n_snaps):
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i==j:
                X_interatomic[n][i][j]=0
            dx_temp = NN_chunks[n][i][1] - NN_chunks[n][j][1]
            dx = dx_temp - a_periodic*int(round(dx_temp/a_periodic))
            dy_temp = NN_chunks[n][i][2] - NN_chunks[n][j][2]
            dy = dy_temp - a_periodic*int(round(dy_temp/a_periodic))
            dz_temp = NN_chunks[n][i][3] - NN_chunks[n][j][3]
            dz = dz_temp - a_periodic*int(round(dz_temp/a_periodic))
            
            X_interatomic[n][i][j]=float(np.sqrt((dx)**2+(dy)**2+(dz)**2))
            
#Computation of cut-off function f_c using Rc
R_c = [4,5,6,7,8]
R_s = [2,3,4,5,6,7,8]
eta = [0.06,0.1,0.2,0.4,1.0]
kappa = [0.5,1.0,1.5,2.0]
zeta = [1,2,4,16,64]
lam = [-1,1]

f_c = np.zeros((n_snaps,n_atoms,n_atoms,len(R_c)))
for n in range(n_snaps):
    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(len(R_c)):
                if X_interatomic[n][i][j]<=R_c[k]:
                    f_c[n][i][j][k] = (np.tanh(1.0 - X_interatomic[n][i][j]/R_c[k]))**3
                    

#Computing radial G vectors using Rc,Rs,eta etc.
G1 = np.sum(f_c,axis=2)
#G2_1, G2_2, G2_3, G2_4 and G2_5 are computed for particular chosen values of eta and Rs
G2_1 = np.sum(np.exp(-eta[0]*(X_interatomic-R_s[0]))*f_c[:,:,:,0],axis=2)
G2_2 = np.sum(np.exp(-eta[1]*(X_interatomic-R_s[1]))*f_c[:,:,:,1],axis=2)
G2_3 = np.sum(np.exp(-eta[2]*(X_interatomic-R_s[2]))*f_c[:,:,:,2],axis=2)
G2_4 = np.sum(np.exp(-eta[3]*(X_interatomic-R_s[3]))*f_c[:,:,:,3],axis=2)
G2_5 = np.sum(np.exp(-eta[4]*(X_interatomic-R_s[4]))*f_c[:,:,:,4],axis=2)

G2_1 = np.reshape(G2_1,(n_snaps,n_atoms,1))
G2_2 = np.reshape(G2_2,(n_snaps,n_atoms,1))
G2_3 = np.reshape(G2_3,(n_snaps,n_atoms,1))
G2_4 = np.reshape(G2_4,(n_snaps,n_atoms,1))
G2_5 = np.reshape(G2_5,(n_snaps,n_atoms,1))

G2 = np.concatenate((G2_1,G2_2,G2_3,G2_4,G2_5),axis=2)
#print("Shape of G1 = "+str(G1.shape))
print("Shape of G2 = "+str(G2.shape))

#Computing angular G functions
cos_theta_ijk = np.zeros((n_snaps,n_atoms,n_atoms,n_atoms))
G4_1 = np.zeros((n_snaps,n_atoms,1))
G4_2 = np.zeros((n_snaps,n_atoms,1))
G4_3 = np.zeros((n_snaps,n_atoms,1))
G4_4 = np.zeros((n_snaps,n_atoms,1))
G4_5 = np.zeros((n_snaps,n_atoms,1))
G4_6 = np.zeros((n_snaps,n_atoms,1))
G4_7 = np.zeros((n_snaps,n_atoms,1))

for n in range(n_snaps):
    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                if j!=i and k!=i and k!=j and X_interatomic[n][i][j]<=R_c[2] and X_interatomic[n][i][k]<=R_c[2]:
                    dx_tempij = NN_chunks[n][i][1] - NN_chunks[n][j][1]
                    dxij = dx_tempij - a_periodic*int(round(dx_tempij/a_periodic))
                    dy_tempij = NN_chunks[n][i][2] - NN_chunks[n][j][2]
                    dyij = dy_tempij - a_periodic*int(round(dy_tempij/a_periodic))
                    dz_tempij = NN_chunks[n][i][3] - NN_chunks[n][j][3]
                    dzij = dz_tempij - a_periodic*int(round(dz_tempij/a_periodic))
                    
                    dx_tempik = NN_chunks[n][i][1] - NN_chunks[n][k][1]
                    dxik = dx_tempik - a_periodic*int(round(dx_tempik/a_periodic))
                    dy_tempik = NN_chunks[n][i][2] - NN_chunks[n][k][2]
                    dyik = dy_tempik - a_periodic*int(round(dy_tempik/a_periodic))
                    dz_tempik = NN_chunks[n][i][3] - NN_chunks[n][k][3]
                    dzik = dz_tempik - a_periodic*int(round(dz_tempik/a_periodic))  
                    
                    Rij = X_interatomic[n][i][j]
                    Rik = X_interatomic[n][i][k]
                    Rjk = X_interatomic[n][j][k]
                    #Rij = np.sqrt(dxij**2 + dyij**2 + dzij**2)
                    #Rik = np.sqrt(dxik**2 + dyik**2 + dzik**2)
                    cos_theta_ijk[n][i][j][k] = (dxij*dxik + dyij*dyik + dzij*dzik)/(Rij*Rik)
                    G4_1[n][i] += (2**(1-zeta[1]))*(((1+lam[0]*cos_theta_ijk[n][i][j][k])**zeta[1]))*(np.exp(-eta[1]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]
                    G4_2[n][i] += (2**(1-zeta[1]))*(((1+lam[1]*cos_theta_ijk[n][i][j][k])**zeta[1]))*(np.exp(-eta[1]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]
                    G4_3[n][i] += (2**(1-zeta[2]))*(((1+lam[0]*cos_theta_ijk[n][i][j][k])**zeta[2]))*(np.exp(-eta[1]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]
                    G4_4[n][i] += (2**(1-zeta[2]))*(((1+lam[1]*cos_theta_ijk[n][i][j][k])**zeta[2]))*(np.exp(-eta[1]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]
                    G4_5[n][i] += (2**(1-zeta[2]))*(((1+lam[0]*cos_theta_ijk[n][i][j][k])**zeta[2]))*(np.exp(-eta[2]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]
                    G4_6[n][i] += (2**(1-zeta[2]))*(((1+lam[1]*cos_theta_ijk[n][i][j][k])**zeta[2]))*(np.exp(-eta[2]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]                    
                    G4_7[n][i] += (2**(1-zeta[3]))*(((1+lam[1]*cos_theta_ijk[n][i][j][k])**zeta[3]))*(np.exp(-eta[3]*(Rij**2 + Rjk**2 + Rik**2)))*f_c[n][i][j][2]*f_c[n][i][k][2]*f_c[n][j][k][2]                    

G4 = np.concatenate((G4_1,G4_2,G4_3,G4_4,G4_5,G4_6,G4_7),axis=2) 
print("Shape of G4 = "+str(G4.shape))
G = np.concatenate((G2,G4),axis=2)
print("Shape of G after concatenation = "+str(G.shape))
print("G2: ",G2)

df = []
df_transformed = []
for i in range(n_snaps):
    temp1 = pd.DataFrame(G[i])
    df.append(temp1)

#MinMax scaling    
scaler = MinMaxScaler()

for i in range(n_snaps):
    temp2 = pd.DataFrame(scaler.fit_transform(df[i]))
    df_transformed.append(temp2)

#df_transformed contains 'n_snaps' normalized dattaframes

print("Data for Snap 0: ",df_transformed[0])
print("\n")

unrolled_list = []
for i in range(n_snaps):
    temp = np.array(df_transformed[i]).reshape((n_atoms*int(G.shape[2])))
    unrolled_list.append(temp)
X_train = np.array(unrolled_list)
y_train = np.array((2.3,4.5,1,2)).reshape(4,1) #energies

#using keras and GridSearch
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import GridSearchCV
def create_model(neurons1,neurons2,neurons3):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_dim=n_atoms*int(G.shape[2]), kernel_initializer='uniform',activation='relu'))
    model.add(Dense(neurons2, kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=50, verbose=True)
param_grid = dict(neurons1=[50,100,150],neurons2=[50,100,150],neurons3=[n_atoms*n_snaps])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, scoring = {'r2','neg_mean_absolute_error','neg_mean_squared_error'},refit='neg_mean_squared_error')
grid_result = grid.fit(X_train, y_train)
print(grid_result.best_params_)
print("Best R2 from GridSearch = "+ str(grid_result.best_score_))

#fitting final model to get weights
model_final = Sequential()
model_final.add(Dense(100, input_dim=n_atoms*int(G.shape[2]), kernel_initializer='uniform',activation='relu'))
model_final.add(Dense(100, kernel_initializer='uniform',activation='relu'))
model_final.add(Dense(n_snaps*n_atoms, kernel_initializer='uniform',activation='relu'))
model_final.add(Dense(1, kernel_initializer='uniform'))
# Compile model
model_final.compile(loss='mean_squared_error', optimizer='adam')
model_final.fit(X_train,y_train)

vec_a = np.array(model_final.get_weights()[5]).reshape((n_snaps*n_atoms,1))
vec_b = np.array(model_final.get_weights()[6])
E_vec = np.multiply(vec_a,vec_b)
print("Shape of E vector is ", E_vec.shape)
print("Energies of {} E vector elements are the individual contributions of atoms in each row".format(E_vec.shape[0]))
print(E_vec)

