import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# Define transforms for data preprocessing
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

## Data DataLoader
# Load the datasets with ImageFolder
data_dir = 'data'
batch_size = 8
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
device = 'cpu'
# Define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Download Pre-trained Resnet50 model. 
model = torchvision.models.resnet50(pretrained=True)

# Replace the last fully-connected layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
device = 'cpu'
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#  Fine-tune the model
for epoch in range(10):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # Print loss and accuracy for each epoch
            if phase == 'train':
                train_loss = running_loss / len(image_datasets[phase])
                train_acc = running_corrects.double() / len(image_datasets[phase])
                print("Epoch {} - Train Loss: {:.4f} Acc: {:.4f}".format(epoch, train_loss, train_acc))
            else:
                val_loss = running_loss / len(image_datasets[phase])
                val_acc = running_corrects.double() / len(image_datasets[phase])
                print("Epoch {} - Validation Loss: {:.4f} Acc: {:.4f}".format(epoch, val_loss, val_acc))

torch.save(model.state_dict(), 'trained_model_1_balanced_data_10_epoch.pth')
torch.save(model, 'trained_model_2_balanced_data_10_epoch.pth')
