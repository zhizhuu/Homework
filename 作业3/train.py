import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GAN_network import *
from ConditionalGAN_network import *
# from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR
from torchstat import stat

# datasets = "facades"
datasets = "cityscapes"
def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        generator (nn.Module): The generator of GAN network model.
        discriminator (nn.Module): The discriminator of GAN network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer_g (Optimizer): Optimizer for updating generator model parameters.
        optimizer_d (Optimizer): Optimizer for updating discriminator model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        # 输入数据
        image_rgb = image_rgb.to(device)
        # 特征图
        image_semantic = image_semantic.to(device)

        # 真实标签全为1，假标签全为0
        real_label = torch.ones(image_semantic.shape[0], 1).to(device)
        fake_label = torch.zeros(image_semantic.shape[0], 1).to(device)

        # 训练判别器
        # Zero the gradients of optimizer_d
        optimizer_d.zero_grad()

        # 生成真假数据送入判别器判断
        noise = torch.randn_like(image_rgb).to(device)
        # 假特征：fake_semantic
        fake_semantic = generator(noise, image_rgb).detach()
        # 真特征：image_semantic
        real_out = discriminator(image_semantic, image_rgb)
        fake_out = discriminator(fake_semantic, image_rgb)

        loss_d_real = criterion(real_out, real_label)
        # loss_d_real.backward()
        loss_d_fake = criterion(fake_out, fake_label)
        # loss_d_fake.backward()
        loss_d = loss_d_real+loss_d_fake
        loss_d.backward()
        optimizer_d.step()
        running_loss += loss_d.item()

        # 训练生成器
        # Zero the gradients of optimizer_g
        optimizer_g.zero_grad()

        # 生成器希望判别器将假数据判别成真数据
        generator_img = generator(noise, image_rgb)
        discriminator_label = discriminator(generator_img, image_rgb)
        loss_g = criterion(discriminator_label, real_label) \
                 + 0.5 * nn.MSELoss()(generator_img, image_semantic)
        loss_g.backward()
        optimizer_g.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, generator_img, 'train_results/'+datasets, epoch)


        # Update running loss
        running_loss += loss_g.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
              f'Loss_g: {loss_g.item():.6f}, '
              f'Loss_d: {loss_d.item():.6f}')

def validate(generator, discriminator, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """

    val_loss_d = 0.0
    val_loss_g = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            # 真实数据
            image_rgb = image_rgb.to(device)
            # 特征图
            image_semantic = image_semantic.to(device)

            # 真实标签全为1，假标签全为0
            real_label = torch.ones(image_semantic.shape[0], 1).to(device)
            fake_label = torch.zeros(image_semantic.shape[0], 1).to(device)

            # 生成真假数据送入判别器判断
            noise = torch.randn_like(image_rgb).to(device)
            # 假特征：fake_semantic
            fake_semantic = generator(noise, image_rgb).detach()
            # 真特征：image_semantic
            real_out = discriminator(image_semantic, image_rgb)
            fake_out = discriminator(fake_semantic, image_rgb)

            loss_d_real = criterion(real_out, real_label)
            loss_d_fake = criterion(fake_out, fake_label)
            loss_d = loss_d_real + loss_d_fake
            val_loss_d += loss_d.item()

            generator_img = generator(noise, image_rgb)
            discriminator_label = discriminator(generator_img, image_rgb)
            loss_g = criterion(discriminator_label, real_label)
            val_loss_g += loss_g.item() + 0.5 * nn.MSELoss()(generator_img, image_semantic)
            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, generator_img, 'val_results/'+datasets, epoch)

    # Calculate average validation loss
    avg_val_loss_g = val_loss_g / len(dataloader)
    avg_val_loss_d = val_loss_d / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}]'
          f',Generate Validation Loss: {avg_val_loss_g:.6f}'
          f',Discriminator Validation Loss: {avg_val_loss_d:.6f}')

# 计算模型的参数个数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    generator = Condition_GAN_Generate(3, 3).to(device)
    # print(generator)
    print("generator parameters:", count_parameters(generator))
    # # print(stat(generator,(3, 256, 256)))
    discriminator = Condition_GAN_Discriminitor(3, 3).to(device)
    # print(discriminator)
    print("discriminator parameters:", count_parameters(discriminator))
    # print(stat(discriminator,(3, 256, 256)))
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_g = StepLR(optimizer_g, step_size=200, gamma=0.2)
    scheduler_d = StepLR(optimizer_d, step_size=128, gamma=0.2)

    # Training loop
    num_epochs = 1600
    for epoch in range(num_epochs):
        train_one_epoch(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epoch, num_epochs)
        validate(generator, discriminator, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_g.step()
        scheduler_d.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/GAN_generate_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/GAN_discriminator_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
