from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np
import torch, os, torchvision
import torchvision.models as models
import time, math
since = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_type = "resnet"
# model_type = "vit"

epochs_num = 100

best_accuracy = 0.0
best_micro_auc = 0.0
best_loss = 10000000.0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        super(Dataset, self).__init__()
        self.path_list = data['path'].tolist()
        self.label_list = data['label'].tolist()
        self.transform = transform

    def __getitem__(self, index):
        path = self.path_list[index]

        tile_file = path
        tile = Image.open(tile_file).convert('RGB')
        data = self.transform(tile)

        label = [self.label_list[index]]
        label = torch.Tensor(label)

        return data, label

    def __len__(self):
        return len(self.path_list)

data = pd.read_csv("./label_int.csv")
transform_ = transforms.Compose([transforms.ToTensor()])
train_dataset = Dataset(data[1400:-21], transform_)
test_dataset = Dataset(data[-21:], transform_)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=14, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=14, shuffle=True, num_workers=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "resnet":
    model = models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)
elif model_type == "vit":
    model = models.vit_b_16(pretrained=True)
    model.heads= torch.nn.Linear(768, 1)

model = model.to(device)
lr = 0.000005
params = model.parameters()
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5, last_epoch=-1, verbose=False)

for epoch in range(epochs_num):
    model.train()
    train_prediction_array = []
    train_label_array = []
    train_possibility_array = []
    train_loss, test_loss = 0, 0
    real_train_loss, real_test_loss = 0,0 

    for data_, label in train_loader:
        data_ = data_.to(device)
        label = label.to(device) #.long()
        label = label.squeeze()
        train_label_array.append(label.cpu().detach().numpy())
        output = torch.nn.functional.softmax(model(data_).squeeze())
        optimizer.zero_grad()
        
        label_ = (label-min(data["label"]))/(max(data["label"])-min(data["label"]))
        # output_ = output*(max(data["label"])-min(data["label"]))+min(data["label"])
        loss = criterion(output, label_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data_.size(0)
        real_train_loss += loss.item() * data_.size(0)*(max(label.cpu().detach().numpy())-min(label.cpu().detach().numpy()))#+min(label.cpu().detach().numpy())

        print("train_loss", loss.item() * data_.size(0))
        
    scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    real_train_loss = real_train_loss/ len(train_loader.dataset)

    with torch.no_grad():
        model.eval()

        test_prediction_array = []
        test_possibility_array = []
        test_label_array = []

        # count = 0

        for data_, label in test_loader:
            data_ = data_.to(device)
            label = label.to(device).long()
            label = label.squeeze()
            test_label_array.append(label.cpu().detach().numpy())
            output = torch.nn.functional.softmax(model(data_).squeeze())
            # output_ = output*(max(data["label"])-min(data["label"]))+min(data["label"])
            label_ = (label-min(data["label"]))/(max(data["label"])-min(data["label"]))
            try:
                loss = criterion(output, label_)
            except:
                print(output.shape)
                print(label.shape)

            test_loss += loss.item() * data_.size(0)
            real_test_loss += loss.item() * data_.size(0)*(max(label.cpu().detach().numpy())-min(label.cpu().detach().numpy()))#+min(label.cpu().detach().numpy())


        test_loss = test_loss / len(test_loader.dataset)
        real_test_loss = real_test_loss / len(test_loader.dataset)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), f"./Models/best_{model_type}_loss_{lr}.pth")
        print("Saving " + "best accuracy " + str(test_loss) + '!!!')

    print('Epoch: {} \tTraining loss: {:.6f}'.format(epoch + 1, train_loss))
    print('Epoch: {} \tTesting loss: {:.6f}'.format(epoch + 1, test_loss))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))