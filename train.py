# %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from model import *
from loss import *
from utils import *


import argparse


def gt_kernel(y):
  label_encoder = LabelEncoder()
  encoded_labels = label_encoder.fit_transform(y)
  encoded_new_label = encoded_labels*2-1
  y = encoded_new_label.reshape(1,-1)
  K = y.T @ y
  # K = np.zeros(len(y)) + np.eye(len(y))
  # for i in range(len(y)):
  #   for j in range(i+1, len(y)):
  #     if y[i] == y[j]:
  #       K[i,j] = K[j,i] = 1
  return K
def multimodal_supervised_kernel(X_list,y,random_state=42):
    K_X = []
    # Z = np.array(range(len(y)))
    indices = np.argsort(y)
    for X in X_list:
        K,K_train, K_val,K_test,Z_ = supervised_kernel(X,y,random_state,indices=indices)
        K_X.append(K)
    K_X = np.stack(K_X)
    print(K_X.shape)
    K_X = torch.from_numpy(K_X)
    # print(Z_.shape,np.array(y).shape) 
    y = np.array(y)[indices]
    y_kernel = gt_kernel(y)
    y_kernel = torch.tensor(y_kernel)
    y_kernel = y_kernel.unsqueeze(dim=0)
    y_kernel = torch.tensor(y_kernel,  dtype=torch.float32)
    K_X = torch.tensor(K_X,  dtype=torch.float32)
    return K_X, y_kernel, Z_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--select_loss_type', type=str, default="mse", help='select_loss_type')
    parser.add_argument('--max_patient', type=int, default=10, help='max_patient')
    parser.add_argument('--saveDir', type=str, default="./figures", help='saveDir')
    parser.add_argument('--dataPathList', type=str, default='./data/AD_CN/GM.csv,./data/AD_CN/PET.csv,./data/AD_CN/CSF.csv', help='dataPathList')
    parser.add_argument('--labelPath', type=str, default='./data/AD_CN/AD_CN_label.csv', help='dataPathList')
    # stop_iter
    parser.add_argument('--stop_iter', type=int, default=500, help='stop_iter')
    
    args = parser.parse_args()
    num_epochs = args.epochs
    max_patient = args.max_patient
    saveDir = args.saveDir
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    dataPathList = [str(item) for item in args.dataPathList.split(',')]
    epochs = args.epochs
    select_loss_type = args.select_loss_type
    nModality = len(dataPathList)
    model = DeepCNN(n_kernel=nModality,kernel_size = 1, n_layer=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    stop_iter = args.stop_iter
    X_list = []
    for dataPath in dataPathList:
        X = load_data(dataPath=dataPath)
        X_list.append(X)
    y = load_data(input=False,dataPath=args.labelPath)
    
    

    best_lost = 999999999999
    evaluation_interval = 1
    df_result = pd.DataFrame(columns=["seed","GM", "PET", "CSF","Summation","Combine"])
    for epoch in range(epochs):
      print("Epoch: ", epoch)
      if epoch > stop_iter:
        stop_iter = 0
        evaluation_interval = 1
        model.eval()
      if epoch%20 == 0 or epoch>stop_iter:
          K_X, K_y, Z = multimodal_supervised_kernel(X_list,y)
          current_seed = epoch
      X_train, y_train = K_X, K_y
      idx = np.where(K_y[0,0,:]==-1)[0][1]
      outputs = model(X_train)
      myloss = my_loss(y_train,outputs,select_loss_type)
      loss =  myloss


      if loss < best_lost:
          if best_lost == 999999999999:
              best_lost = loss
              continue
          print(f"***************loss < best loss: {loss} < {best_lost}*****************,saving")
          best_lost = loss
      if epoch < stop_iter:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      print(f"epoch = {epoch}, loss: {loss},  current_seed:{current_seed}, select_loss_type:{select_loss_type}")


      if epoch%evaluation_interval == 0:
        print(f"========= Combine ==============")
        acc = classify_kernel(outputs[0,:,:].detach().numpy(),y_train,Z)
        list_modelity = ["GM","MRI", "PET", "CSF", "SNP"]
        modelity_acc =[]
        best_acc = 0
        for j in range(nModality+1):
            if j<nModality:
              print(f"========= {list_modelity[j]} ==============")
              acc_i = classify_kernel(X_train.detach().numpy()[j,:,:],y_train,Z)
            else:
              acc_i = classify_kernel( torch.sum(X_train,axis=0).detach().numpy(),y_train,Z)
              print("acci",acc_i)
            if acc_i > best_acc:
              best_acc = acc_i
            modelity_acc.append(acc_i)
        df_result.loc[len(df_result)] =[current_seed]+ modelity_acc + [acc]
        df_result.to_csv(os.path.join(saveDir,'result.csv'),index=False)

        plt.figure(figsize=(15,10))
        
        for idx in range(nModality):
          plt.subplot(3,3,1)
          plt.imshow(y_train.detach().numpy()[0,:,:])
          plt.xticks([idx], "-")
          plt.yticks([idx], "-")

          plt.subplot(3,3,2)
          plt.imshow(outputs.detach().numpy()[0,:,:])
          plt.xticks([idx], "-")
          plt.yticks([idx], "-")
          plt.title(f"{acc*100:.2f}")

          plt.subplot(3,3,3)
          plt.imshow(torch.sum(X_train,axis=0).detach().numpy())
          plt.xticks([idx], "-")
          plt.yticks([idx], "-")
          plt.title(f"{modelity_acc[-1]*100:.2f}")


          plt.subplot(3,3,4+idx)
          plt.imshow(X_train.detach().numpy()[idx,:,:])
          plt.xticks([idx], "-")
          plt.yticks([idx], "-")
          plt.title(f"{modelity_acc[idx]*100:.2f}")
          plt.savefig(os.path.join(saveDir,f"{select_loss_type}_{current_seed}_{epoch}.png"))

          # plt.show()