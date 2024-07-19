import torch
import torch.nn as nn
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset
import random
import os

class NeuralNetwork(nn.Module):
    #2 Layers i want it to be from input size to output size
    def __init__(self, input_size,hidden_dim, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_dim)
        self.ll = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.hidden_size = hidden_dim
        self.num_layers = 1

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.ll(x)
        x = self.softmax(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.log_softmax(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, batch_size):
    for epoch in range(num_epochs):
        model.train()
        for (mfccs, labels) in train_loader:
            # Prepare input and target lengths
            mfccs = mfccs.squeeze(0)
            labels = labels.squeeze(0)
            input_lengths = torch.full((batch_size,), mfccs.size(1), dtype=torch.long)
            target_lengths = torch.full((batch_size,), labels.size(0), dtype=torch.long)

            # Forward pass
            outputs = model(mfccs)

            # Compute CTC loss
            loss = criterion(outputs, labels, input_lengths, target_lengths)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                input_lengths = torch.full((batch_size,), mfccs.size(1), dtype=torch.long)
                target_lengths = torch.full((batch_size,), labels.size(1), dtype=torch.long)
                outputs = model(mfccs)
                val_loss += criterion(outputs, labels, input_lengths, target_lengths).item()

        print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss / len(val_loader)}')


def load_wav_files(base_dir, n_mfcc):
    digit_to_int = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }

    fixed_length = 16000
    mfcc_features = []
    labels = []
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for digit in digits:
        digit_dir = os.path.join(base_dir, str(digit))
        for filename in os.listdir(digit_dir):
            filepath = os.path.join(digit_dir, filename)
            if (not filename.endswith(".wav")):
                continue
            wav, sr = librosa.load(filepath, sr=None)
            wav = librosa.util.fix_length(wav, size=fixed_length)
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
            mfcc_features.append(mfcc)
            labels.append(digit_to_int[digit])

    return mfcc_features, labels

class DigitSequenceDataset(Dataset):
    def __init__(self, mfcc_features, labels, sequence_length=5):
        self.mfcc_features = mfcc_features
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.mfcc_features) // self.sequence_length

    def __getitem__(self, idx):
        mfcc_seq = []
        label_seq = []
        for _ in range(self.sequence_length):
            rand_idx = random.randint(0, len(self.mfcc_features) - 1)
            mfcc_seq.append(self.mfcc_features[rand_idx])
            label_seq.append(self.labels[rand_idx])
        mfcc_seq = np.concatenate(mfcc_seq, axis=1)
        return torch.tensor(mfcc_seq, dtype=torch.float32), torch.tensor(label_seq, dtype=torch.long)

def q3():
    n_mfcc = 13
    train_mfcc, train_labels = load_wav_files("train", n_mfcc)
    val_mfcc, val_labels = load_wav_files("val", n_mfcc)
    test_mfcc, test_labels = load_wav_files("test", n_mfcc)

    train_dataset = DigitSequenceDataset(train_mfcc, train_labels, sequence_length=8)
    val_dataset = DigitSequenceDataset(val_mfcc, val_labels, sequence_length=8)
    test_dataset = DigitSequenceDataset(test_mfcc, test_labels, sequence_length=8)

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size,drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, drop_last=True, shuffle=False)

    criterion = nn.CTCLoss()
    model = NeuralNetwork(n_mfcc, 128, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size)


if __name__ == "__main__":
    q3()