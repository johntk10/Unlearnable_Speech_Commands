import torch
from tqdm import tqdm
from torchaudio import save

def noise_generator(num_samples, noise_shape, noise_factor):
    # Increase the noise factor to make the examples harder to learn
    noise = noise_factor * torch.randn(num_samples, *noise_shape)
    return noise

def calculate_accuracy(model, data_loader, noise, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device) + noise[:len(images)].to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def generate_unlearnable_examples(base_model, clean_train_loader, criterion, optimizer, noise_generator, device, epochs):
    noise_factor = 1.0  # Increase noise factor to make examples harder to learn
    noise = noise_generator(50000, (1, 16000), noise_factor).to(device)
    data_iter = iter(clean_train_loader)
    train_idx = 0

    for epoch in tqdm(range(epochs)):  # 10 epochs (adjust as needed)
        base_model.train()
        for param in base_model.parameters():
            param.requires_grad = True

        for j in range(10):  # 10 iterations per epoch (adjust as needed)
            try:
                images, labels = next(data_iter)
            except StopIteration:
                train_idx = 0
                data_iter = iter(clean_train_loader)
                images, labels = next(data_iter)

            images = images.to(device) + noise[train_idx].to(device)
            train_idx += 1

            labels = labels.to(device)
            base_model.zero_grad()
            optimizer.zero_grad()
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
            optimizer.step()

            accuracy = calculate_accuracy(base_model, clean_train_loader, noise, device)
            print(f'Epoch {epoch+1}, Iteration {j+1}, Accuracy: {accuracy*100:.4f}%')

    return noise
