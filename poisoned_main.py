import torch
import torch.optim as optim
import torchaudio
from tqdm import tqdm
from dataset import SubsetSC, format_data, dataloaders
from net import M5
from train import train, test
from unlearnable_examples import generate_unlearnable_examples, noise_generator

NEW_SAMPLE_RATE = 8000
BATCH_SIZE = 256
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
STEP_SIZE = 20
GAMMA = 0.1
LOG_INTERVAL = 20
EPOCHS = 10

if __name__ == '__main__':
    trainset = SubsetSC('training')
    testset = SubsetSC('testing')

    waveform, sample_rate, label, speaker_id, utterance_number = trainset[0]
    labels = sorted(list(set(datapoint[2] for datapoint in trainset)))
    transformed, transform = format_data(NEW_SAMPLE_RATE, sample_rate, waveform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    trainloader, testloader = dataloaders(trainset, testset, BATCH_SIZE, num_workers, pin_memory)

    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    pbar_update = 1 / (len(trainloader) + len(testloader))
    losses = []

    transform = transform.to(device)

    # Generate unlearnable examples noise
    noise = generate_unlearnable_examples(model, trainloader, torch.nn.CrossEntropyLoss(), optimizer, noise_generator, device, EPOCHS)

    '''with tqdm(total=EPOCHS) as pbar:
        for epoch in range(1, EPOCHS + 1):
            train(device, model, optimizer, pbar_update, epoch, LOG_INTERVAL, trainloader, transform, pbar, losses)
            test(device, model, pbar_update, epoch, testloader, transform, pbar)
            scheduler.step()
        # torch.save(model.state_dict(), 'model.pth')'''

    torch.save(noise, 'unlearnable_noise.pth')
