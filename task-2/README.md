```python
import kagglehub
dataset_path = kagglehub.dataset_download('bhavikjikadara/dog-and-cat-classification-dataset')
print('Data source import complete.')

```

    Data source import complete.
    


```python
import os
directory = os.path.join(dataset_path, 'PetImages')
images = []
labels = []
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    if os.path.isdir(folder_path):
        file_count = 0
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.jpg')):
                images.append(file_path)
                labels.append(folder)
                file_count += 1
```


```python
import pandas as pd

df = pd.DataFrame({
    'image_path': images,
    'label': labels
    })

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24998 entries, 0 to 24997
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   image_path  24998 non-null  object
     1   label       24998 non-null  object
    dtypes: object(2)
    memory usage: 390.7+ KB
    


```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

# 0 - cat, 1 - dog
df['label_class'] = df['label'].apply(lambda x: 0 if x=='Cat' else 1)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_class'], random_state=42)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_imgs(df):
    imgs = []
    labels = []
    for i, row in df.iterrows():
        img = Image.open(row['image_path']).convert('RGB')
        img = transform(img)
        imgs.append(img)
        labels.append(row['label_class'])
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    return TensorDataset(imgs, labels)


train_dataset = load_imgs(train_df)
test_dataset  = load_imgs(test_df)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=32)

```

    c:\Users\Анна\AppData\Local\Programs\Python\Python313\Lib\site-packages\PIL\TiffImagePlugin.py:950: UserWarning: Truncated File Read
      warnings.warn(str(msg))
    


```python
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) #сверточный слой
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) #Слой субдискретизации
        self.fullc1 = nn.Linear(8192, 64) #полносвязный слой 
        self.fullc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8192)
        x = torch.relu(self.fullc1(x))
        x = self.fullc2(x)
        return x

model = CNN(num_classes=2)
criterion = nn.CrossEntropyLoss() #функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001) #Оптимизатор
```


```python
#обучение модели
num_epochs = 4
for epoch in range(num_epochs):
    for imgs, labels in train_loader:
        outputs = model(imgs)
        # Вычисление потерь
        loss = criterion(outputs, labels)
        optimizer.zero_grad()  #обнул. градиенты
        loss.backward()  # вычисление градиентов
        optimizer.step()  #обнов параметры
```


```python
from sklearn.metrics import accuracy_score, f1_score

pred_labels=[]
true_labels = []
for imgs, labels in train_loader:
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1)
    pred_labels.extend(preds.numpy())
    true_labels.extend(labels.numpy())

print("Train: Accuracy:", accuracy_score(true_labels, pred_labels))
print("F1:", f1_score(true_labels, pred_labels))

pred_labels=[]
true_labels = []
for imgs, labels in test_loader:
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1)
    pred_labels.extend(preds.numpy())
    true_labels.extend(labels.numpy())

print("Test: Accuracy:", accuracy_score(true_labels, pred_labels))
print("F1:", f1_score(true_labels, pred_labels))
```

    Train: Accuracy: 0.8973897389738974
    F1: 0.8954341622503057
    Test: Accuracy: 0.816
    F1: 0.8099173553719008
    
