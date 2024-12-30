import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet50 import create_resnet50
from torch.utils.data import DataLoader, Subset
import random
import os
from utils import get_device
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import kagglehub


# Training configuration
class Config:
    batch_size = 64
    num_epochs = 3
    learning_rate = 0.001
    num_classes = 1000
    subset_size = 1000  # Small subset for local training
    device = get_device()

class Data_Metrics:
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []


def create_data_loaders(train_data_dir, test_data_dir):
    # Define data transformations
    train_transformation = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transformation)
    test_dataset = datasets.ImageFolder(test_data_dir, transform=test_transform)
    
    is_training_locally = False
    # Create data loader
    if is_training_locally:
        # Create a small subset for local training
        train_indices = random.sample(range(len(train_dataset)), Config.subset_size)
        train_subset_dataset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(
            train_subset_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=10
        )
        test_indices = random.sample(range(len(test_dataset)), Config.subset_size)
        test_subset_dataset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(
            test_subset_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=10
        )
    else:
       train_loader = DataLoader(
            train_dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=10
        )
       test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=10) 
    
    print(f"[INFO] Total training batches: {len(train_loader)}")
    print(f"[INFO] Batch size: {Config.batch_size}")
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Test samples: {len(test_dataset)}\n")
    return train_loader,test_loader

def train_and_test_model():
    print(f"Device {Config.device}")

    print("[STEP 1] Preparing datasets...")
    # Download latest version
    path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
    path += "/imagenet-mini"
    # path = "/Users/gitesh.grover/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini"
    print("Path to dataset files:", path)
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'val') 
    # Load data
    train_loader, test_loader = create_data_loaders(train_dir, test_dir)

# Initialize model
    print("[STEP 2] Initializing model...")
    # Initialize model
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, Config.num_classes)
    # model = model.to(Config.device)
    # Initialize model
    model = create_resnet50(num_classes=Config.num_classes)
    model = model.to(Config.device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = Config.num_epochs * steps_per_epoch
    
    # Replace StepLR with OneCycleLR
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.1)    
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.1,              # Maximum learning rate
    #     total_steps=total_steps,
    #     pct_start=0.3,           # Peak at 30% of training
    #     div_factor=10,           # Initial lr = max_lr/div_factor
    #     final_div_factor=10000,   # Final lr = max_lr/final_div_factor
    #     anneal_strategy='cos'    # Cosine annealing
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.001, threshold_mode='abs', eps=0.001, verbose=True)


    print("[STEP 3] Starting training and Testing...")
    # Training loop
    start_time = time.time()
    for epoch in range(Config.num_epochs):
        print(f"\n[INFO] Training of Epoch {epoch+1} started...")
        train_model(model, train_loader, optimizer, scheduler, criterion, epoch)
        training_time = time.time() - start_time
        print(f"[INFO] Training of Epoch {epoch+1} completed in {training_time:.2f} seconds")

        print("[INFO] Evaluating model...")
        # scheduler.step()
        scheduler.step(Data_Metrics.train_accuracies[-1]*.01)
        print("Current learning rate:", scheduler.get_last_lr()[0])
        test_model(model, test_loader, Config.device)
    
    # Save the model  
    print("[STEP 4] Saving the model...")
    torch.save(model.state_dict(), 'resnet50_local_trained.pth') 

    # Plot the training and testing losses
    print("\n[STEP 5] Plot the training and testing losses...")
    printLossAndAccuracy()
        
    print('Training completed!')
    return model

#Train this epoch
def train_model(model, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(Config.device)
        labels = labels.to(Config.device)
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when we start our training loop, we should zero out the gradients so that the parameter update is correct.
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        loss = criterion(outputs, labels)
         # loss = nn.functional.nll_loss(output, target)
        Data_Metrics.train_losses.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        # scheduler.step() Only step when need to step in between of a running epoch

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar every batch
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'accuracy': f'{accuracy:.2f}%'
        })
        Data_Metrics.train_accuracies.append(100*correct/total)

def test_model(model, test_loader, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)
            
            #loss = nn.CrossEntropyLoss()(output, target, reduction='sum')
            loss = nn.CrossEntropyLoss()(output, labels)
            running_loss += loss.cpu().item() # TODO whether it needs to be cpu
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        running_loss /= len(test_loader.dataset)
        Data_Metrics.test_losses.append(running_loss)
        final_accuracy = 100. * correct / total
        print(f"Test Accuracy: {final_accuracy:.2f}%")
        Data_Metrics.test_accuracies.append(final_accuracy)

def printLossAndAccuracy():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(Data_Metrics.train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(Data_Metrics.train_accuracies)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(Data_Metrics.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(Data_Metrics.test_accuracies)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

def save_checkpoint(model, epoch, optimizer, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

if __name__ == '__main__':
    model = train_and_test_model()