import torch, os, cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
])

def training_loop_supervised(model, train_loader, test_loader, num_epochs,
                             criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch image classification model, saving the model with the best accuracy on the test dataset.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the testing data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best accuracy on the test set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - test_accuracy_history: List of test accuracies for each epoch.
    """
    print("HAAAAAAAAAAA")
    model.to(device)
    train_loss_history = []
    test_accuracy_history = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        test_accuracy_history.append(test_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_train_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2%}')

        # Save the model if it has the best accuracy on the test set
        if test_accuracy == max(test_accuracy_history):
            torch.save(model.state_dict(), model_path)

    print('Training finished!')

    # Code here
    print("train_loss_history", train_loss_history)
    print("test_accuracy_history", test_accuracy_history)
    return train_loss_history, test_accuracy_history


def training_loop_unsupervised(model, task, train_loader, test_loader, num_epochs,
                             criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch image classification model, saving the model with the best accuracy on the test dataset.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - task: string, either 'reconstruction' or 'denoising'
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the testing data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best accuracy on the test set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - test_loss_history: List of test losses for each epoch.
    """
    model.to(device)
    train_loss_history = []
    test_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            if task == 'denoising':
                noisy_imgs = add_gaussian_noise(images)  # Add Gaussian noise
                outputs = model(noisy_imgs)
            elif task == 'reconstruction':
                outputs = model(images)
            else:
                raise ValueError("Invalid task! Choose 'reconstruction' or 'denoising'.")

            loss = criterion(outputs, images)  # Calculate reconstruction loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Evaluate on the test set
        model.eval()
        total_test_loss = 0.0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                if task == 'denoising':
                    noisy_imgs = add_gaussian_noise(images)  # Add Gaussian noise
                    outputs = model(noisy_imgs)
                elif task == 'reconstruction':
                    outputs = model(images)
                else:
                    raise ValueError("Invalid task! Choose 'reconstruction' or 'denoising'.")

                # Calculate reconstruction loss
                loss = criterion(outputs, images)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}')

        # Save the model if it has the best test loss
        if avg_test_loss == min(test_loss_history):
            torch.save(model.state_dict(), model_path)

    print('Training finished!')

    return train_loss_history, test_loss_history



def noisy_images(images, mean=0.0, std=0.5):
    """
    Adds Gaussian noise to a batch of images.

    Parameters:
    - images: tensor of images of shape (N, C, H, W), where N is the batch size,
              C is the number of channels, H is the height, and W is the width.
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.

    Returns:
    - Tensor of noisy images of the same shape as the input.
    """
    # Code here
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    return noisy_images


def add_gaussian_noise(images, mean=0.0, std=0.5):
    """
    Adds Gaussian noise to a batch of images.

    Parameters:
    - images: tensor of images of shape (N, C, H, W), where N is the batch size,
              C is the number of channels, H is the height, and W is the width.
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.

    Returns:
    - Tensor of noisy images of the same shape as the input.
    """
    # Generate Gaussian noise with the same shape as the input images
    noise = torch.randn_like(images) * std + mean

    # Add the noise to the input images
    noisy_images = images + noise

    # Clamp the pixel values between 0 and 1
    noisy_images = torch.clamp(noisy_images, min=0.0, max=1.0)

    return noisy_images

def visualize_reconstructions(test_loader, model, device, task='reconstruction'):
    """
    Visualizes original and reconstructed images from a test dataset.

    Args:
    - test_loader: DataLoader for the test dataset.
    - model: Trained model for image reconstruction or denoising.
    - device: The device (e.g., 'cuda' or 'cpu') the model is running on.
    - task: The task to perform - either 'reconstruction' or 'denoising'.
    """
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    if task == 'denoising':
        images = noisy_images(images)  # Apply noise if the task is denoising

    recons = model(images.to(device))

    images_np = images.cpu().numpy()
    recons_np = recons.cpu().detach().numpy()

    indices = np.random.choice(images_np.shape[0], 9, replace=False)
    selected_images = images_np[indices]
    selected_recons = recons_np[indices]
    selected_labels = labels.numpy()[indices]

    fig, axs = plt.subplots(2, 9, figsize=(15, 4))

    for i in range(9):
        axs[0, i].imshow(np.transpose(selected_images[i], (1, 2, 0)), interpolation='none', cmap='gray')
        axs[0, i].set_title(f"GT: {selected_labels[i]}")
        axs[0, i].axis('off')

        axs[1, i].imshow(np.transpose(selected_recons[i], (1, 2, 0)), interpolation='none', cmap='gray')
        axs[1, i].set_title(f"Recon: {selected_labels[i]}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()




# Dataset class for Loading Unlabeled Images:
class UnlabeledImageDataset(Dataset):
    def __init__(self, folder, split='train', split_ratio=0.8):
        self.folder = folder
        self.imgs = os.listdir(folder)
        self.split = split
        self.split_ratio = split_ratio
        
        num_imgs = len(self.imgs)
        self.split_index = int(num_imgs * split_ratio)
        
        if split == 'train':
            self.imgs = self.imgs[:self.split_index]
        elif split == 'val':
            self.imgs = self.imgs[self.split_index:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.imgs[idx])
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pt = Image.fromarray(img_np)  # Convert numpy array to PIL Image
        img_pt = transform(img_pt)  # Apply transformations
        return img_pt, 0    # adding a placeholder lablel
 
class UnlabeledImageDataset_2(Dataset):
    def __init__(self, folder, split='train', split_ratio=0.8):
        self.folder = folder
        self.transform = transform
        all_imgs = os.listdir(folder)
        num_imgs = len(all_imgs)
        split_index = int(num_imgs * split_ratio)

        if split == 'train':
            self.imgs = all_imgs[:split_index]
        elif split == 'val':
            self.imgs = all_imgs[split_index:]

        self.images = []
        for img_name in tqdm(self.imgs):
            img_path = os.path.join(self.folder, img_name)
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img_np = self.transform(img_np)
                
            self.images.append(img_np)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # since images are already preloaded, just retrieve from the list
        return self.images[idx]
    
def unlabeled_dataset(ram_above_16: bool = False):
    if ram_above_16:
        return UnlabeledImageDataset_2
    else:
        return UnlabeledImageDataset

def training_loop_segmentation(model, train_loader, val_loader, num_epochs,
                               criterion, optimizer, device, model_path):
    """
    Trains and evaluates a PyTorch segmentation model, saving the model with the best performance on the validation dataset.
    Stores 10 sample outputs every 10 epochs.

    Args:
    - model: The PyTorch model to be trained and evaluated.
    - train_loader: DataLoader for the training data.
    - val_loader: DataLoader for the validation data.
    - num_epochs: Number of epochs to train the model.
    - criterion: Loss function used for training.
    - optimizer: Optimization algorithm used for training.
    - device: The device ('cuda' or 'cpu') to perform training and evaluation on.
    - model_path: Path to save the model achieving the best performance on the validation set.

    Returns:
    - train_loss_history: List of average training losses for each epoch.
    - val_loss_history: List of validation losses for each epoch.
    - sample_outputs: List of dictionaries containing images, predicted masks, and ground truth masks for sample outputs.
    """

    model.to(device)
    train_loss_history = []
    val_loss_history = []
    sample_outputs = []
    best_val_loss = float('inf')  # Initialize best_val_loss with a large value
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        i=0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images.float())  # Convert input to float
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            if i%500==0:
                print("Han bhai chal ra hai",i)
            i+=1
        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        if epoch % 100 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images.float())  # Convert input to float
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0) #might have to remove images.size(0)
                    if len(sample_outputs) < 10:
                        sample_outputs.append({
                            'image': images[0],
                            'predicted_mask': outputs[0],
                            'ground_truth_mask': masks[0]
                        })
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_history.append(val_loss)

            if len(sample_outputs) == 10:
                print(f'Saving sample outputs for epoch {epoch}...')
                torch.save(sample_outputs, f'sample_outputs_epoch_{epoch}.pt')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    return train_loss_history, val_loss_history, sample_outputs