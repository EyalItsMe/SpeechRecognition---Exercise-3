import math

import torch
import torch.nn as nn
import numpy as np
import librosa
from torch.utils.data import DataLoader, Dataset
import random
import os
import matplotlib.pyplot as plt

digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class NeuralNetwork(nn.Module):
    # 2 Layers I want it to be from input size to output size
    def __init__(self, input_size, hidden_dim, output_size=27):
        super(NeuralNetwork, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, num_layers=1, batch_first=True)
        self.ll = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.hidden_size = hidden_dim

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.ll(x)
        x = self.softmax(x)
        return x



def create_digit_to_int_mapping():
    digit_to_int = {}

    for word in digits:
        int_list = [ord(char) - ord('a') for char in word]
        digit_to_int[word] = int_list

    return digit_to_int


def train(model, train_loader, val_loader,criterion, optimizer, num_epochs, batch_size):
    train_losses = []
    val_losses = []
    digit_to_int_mapping = create_digit_to_int_mapping()
    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        for (mfccs, labels) in train_loader:
            # Prepare input and target lengths
            mfccs = mfccs.permute(0, 2, 1)  # Reshape for RNN input
            input_lengths = torch.full((batch_size,), mfccs.size(2), dtype=torch.long)
            lengths = [len(s) for s in labels]
            target_lengths = torch.tensor(lengths, dtype=torch.long)

            int_labels = [digit_to_int_mapping[label] for label in labels]
            labels = [item for sublist in int_labels for item in sublist]
            labels = torch.tensor(labels, dtype=torch.long)
            # Forward pass
            outputs = model(mfccs)
            outputs = outputs.permute(1, 0, 2)  # Reshape for CTC loss
            loss = criterion(outputs, labels, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping NaN/Inf loss at epoch {epoch + 1}")
                continue
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs = mfccs.permute(0, 2, 1)
                input_lengths = torch.full((batch_size,), mfccs.size(1), dtype=torch.long)
                lengths = [len(s) for s in labels]
                target_lengths = torch.tensor(lengths, dtype=torch.long)
                int_labels = [digit_to_int_mapping[label] for label in labels]
                int_labels = [item for sublist in int_labels for item in sublist]
                int_labels = torch.tensor(int_labels, dtype=torch.long)
                outputs = model(mfccs)
                min_loss = [math.inf] * len(labels)
                max_digits = [""] * len(labels)
                for digit in digits:
                    for i, num in enumerate(min_loss):
                        current_num = torch.tensor(digit_to_int_mapping[digit], dtype=torch.long)
                        loss = criterion(outputs[i], current_num,
                                         input_lengths[i], torch.tensor(current_num.size(0)))
                        if loss < min_loss[i]:
                            max_digits[i] = digit
                            min_loss[i] = loss
                for i, max_digit in enumerate(max_digits):
                    if max_digit == labels[i]:
                        accuracy += 1
                outputs = outputs.permute(1, 0, 2)
                val_loss += criterion(outputs, int_labels, input_lengths, target_lengths).item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        print("Validation accuracy is " + str(accuracy) + "%")
    return train_losses, val_losses

def test_model(test_loader, batch_size,model, criterion):
    model.eval()
    test_loss = 0
    accuracy = 0
    digit_to_int_mapping = create_digit_to_int_mapping()
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.permute(0, 2, 1)
            input_lengths = torch.full((batch_size,), mfccs.size(1), dtype=torch.long)
            lengths = [len(s) for s in labels]
            target_lengths = torch.tensor(lengths, dtype=torch.long)
            int_labels = [digit_to_int_mapping[label] for label in labels]
            int_labels = [item for sublist in int_labels for item in sublist]
            int_labels = torch.tensor(int_labels, dtype=torch.long)
            outputs = model(mfccs)
            min_loss = [math.inf] * len(labels)
            max_digits = [""] * len(labels)
            for digit in digits:
                for i, num in enumerate(min_loss):
                    current_num = torch.tensor(digit_to_int_mapping[digit], dtype=torch.long)
                    loss = criterion(outputs[i], current_num,
                                     input_lengths[i], torch.tensor(current_num.size(0)))
                    if loss < min_loss[i]:
                        max_digits[i] = digit
                        min_loss[i] = loss
            for i, max_digit in enumerate(max_digits):
                if max_digit == labels[i]:
                    accuracy += 1
            outputs = outputs.permute(1, 0, 2)
            test_loss += criterion(outputs, int_labels, input_lengths, target_lengths).item()

    test_loss = test_loss / len(test_loader)
    accuracy = accuracy / len(test_loader)
    print("Test loss is " + str(test_loss))
    print("Test accuracy is " + str(accuracy) + "%")
def load_wav_files(base_dir, n_mfcc):
    fixed_length = 16000
    mfcc_features = []
    labels = []
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
            labels.append(digit)

    return mfcc_features, labels


class DigitSequenceDataset(Dataset):
    def __init__(self, mfcc_features, labels, sequence_length=5):
        self.mfcc_features = mfcc_features
        self.labels = labels

    def __len__(self):
        return len(self.mfcc_features)

    def __getitem__(self, idx):
        mfcc = self.mfcc_features[idx]
        label = self.labels[idx]
        return torch.tensor(mfcc, dtype=torch.float32), label


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def q3():
    n_mfcc = 20
    train_mfcc, train_labels = load_wav_files("train", n_mfcc)
    val_mfcc, val_labels = load_wav_files("val", n_mfcc)
    test_mfcc, test_labels = load_wav_files("test", n_mfcc)

    train_dataset = DigitSequenceDataset(train_mfcc, train_labels)
    val_dataset = DigitSequenceDataset(val_mfcc, val_labels)
    test_dataset = DigitSequenceDataset(test_mfcc, test_labels)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    criterion = nn.CTCLoss()
    model = NeuralNetwork(n_mfcc, 256, 27)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=15,
                                 batch_size=batch_size)
    test_model(test_loader,batch_size, model, criterion)
    plot_loss(train_loss, val_loss)


if __name__ == "__main__":
    q3()
