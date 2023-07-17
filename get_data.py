pic_path = "/content/pic/"
!pip install pyts
import pyts
import torchvision
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
from pandas import DataFrame
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np


if not os.path.exists(pic_path):
  os.makedirs(pic_path)
def sliding_window(data, window_size=15):
  """
  sliding window to get the data for train and test
  """
  generator = TimeseriesGenerator(data, data, length=window_size, batch_size=1, stride=1)
  X_train, y_train = [], []
  for i in range(len(generator)):
    x, y = generator[i]
    X_train.append(x[0])
    y_train.append(y[0])
  return X_train, y_train

def to_img(data, window_size):
    from pyts.image import GramianAngularField
    gaf = GramianAngularField(image_size=window_size)
    gaf_images = gaf.fit_transform(data.T)
    gaf_tensor = torch.tensor(gaf_images).float()
    # print(gaf_tensor.shape)
    y = F.interpolate(gaf_tensor.unsqueeze(0), scale_factor=16, mode='bilinear', align_corners=False)
    img_name = pic_path+str(len(os.listdir(pic_path)))+".jpg"
    torchvision.utils.save_image(y.squeeze(0),img_name)
    # return gaf_images


from pandas_datareader import data
import yfinance as yf
yf.pdr_override()
apple = data.get_data_yahoo(tickers='AAPL', start='2012-01-01', end='2023-12-13')
data = apple['Close'].to_numpy()
window_size = 14
X_train, y_train = sliding_window(data, window_size=window_size)

i = 0
import pandas as pd
df = pd.DataFrame(columns=['path','label'])
for i, x_train in enumerate(X_train):
  #  if i ==2:
  #     break
  #  i = i+1
   to_img(x_train.reshape(-1,1), window_size)
   path_i = "/content/pic/"+str(len(os.listdir(pic_path)))+".jpg"
   df.loc[len(df.index)] = [path_i, y_train[i]]

df["label"] = df["label"].astype('int')
df.to_csv("./label.csv", index=False)
print("done")