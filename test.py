from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np
import torch, os, torchvision
import torchvision.models as models
import time, math
since = time.time()
from pylab import plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_type = "resnet"
# model_type = "vit"

best_accuracy = 0.0
best_micro_auc = 0.0
best_loss = 10000000.0
look_back = 50
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
train_dataset = Dataset(data[:-21], transform_)
test_dataset = Dataset(data[-21:], transform_)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=14, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=14, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == "resnet":
    model = models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)
elif model_type == "vit":
    model = models.vit_b_16(pretrained=True)
    model.heads= torch.nn.Linear(768, 1)

model = model.to(device)
checkpoint = torch.load("/home/r15user12/code/7409/Models/best_resnet_loss_5e-06.pth")
model.load_state_dict(checkpoint)
lr = 0.000005
params = model.parameters()
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5, last_epoch=-1, verbose=False)

for epoch in range(1):
    train_loss, test_loss = 0, 0
    real_train_loss, real_test_loss = 0,0 

    result = []

    with torch.no_grad():
        model.eval()

        test_prediction_array = []
        test_possibility_array = []
        test_label_array = []

        for data_, label in test_loader:
            data_ = data_.to(device)
            label = label.to(device).long()
            label = label.squeeze()
            test_label_array.append(label.cpu().detach().numpy())
            output = torch.nn.functional.softmax(model(data_).squeeze())
            label_ = (label-min(data["label"]))/(max(data["label"])-min(data["label"]))
            result += list(output.cpu().detach().numpy()*(max(label.cpu().detach().numpy())-min(label.cpu().detach().numpy()))+min(label.cpu().detach().numpy()))
            loss = criterion(output, label_)

            test_loss += loss.item() * data_.size(0)
            real_test_loss += loss.item() * data_.size(0)*(max(label.cpu().detach().numpy())-min(label.cpu().detach().numpy()))+min(label.cpu().detach().numpy())


        test_loss = test_loss / len(test_loader.dataset)
        real_test_loss = real_test_loss / len(test_loader.dataset)

    with open("./test.txt", "w") as f:
        f.write(str(result).replace("[","").replace("]","").replace(", ",","))

    predict_data = np.array(result)
    label = np.array(data["label"][-21-(look_back):].tolist())
    plt.plot(label)
    testPredictPlot=np.empty_like(label)
    testPredictPlot[:-21] = np.array(label[:-21].tolist())
    testPredictPlot[-21:] = np.array(predict_data)+2
    plt.plot(testPredictPlot)
    plt.legend(['true price', 'predict price'])
    plt.savefig("test.png")
