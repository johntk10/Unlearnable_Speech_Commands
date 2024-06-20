import torch
import torch.nn.functional as F

def train(device, model, optimizer, pbar_update, epoch, log_interval, trainloader, transform, pbar, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        data = transform(data)
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}")
            pbar.update(pbar_update)
            losses.append(loss.item())

def test(device, model, pbar_update, epoch, testloader, transform, pbar):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            data = transform(data)
            output = model(data)
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
            pbar.update(pbar_update)
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(testloader.dataset)} ({100. * correct / len(testloader.dataset):.0f}%)\n")

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)
