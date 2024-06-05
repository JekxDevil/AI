import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_imgs(path, folders):
    imgs = []
    labels = []
    n_imgs = 0
    for c in folders:
        # iterate over all the files in the folder
        for f in os.listdir(os.path.join(path, c).replace("/mnt/z", 'Z:\\')):
            if not f.endswith('.jpg'):
                continue
            # load the image (here you might want to resize the img to save memory)
            im = Image.open(os.path.join(path, c, f)).copy()
            imgs.append(im)
            labels.append(c)
        print('Loaded {} images of class {}'.format(len(imgs) - n_imgs, c))
        n_imgs = len(imgs)
    print('Loaded {} images total.'.format(n_imgs))
    return imgs, labels


def plot_sample(imgs, labels, nrows=4, ncols=4, resize=None):
    # create a grid of images
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # take a random sample of images
    indices = np.random.choice(len(imgs), size=nrows * ncols, replace=False)
    for ax, idx in zip(axs.reshape(-1), indices):
        ax.axis('off')
        # sample an image
        ax.set_title(labels[idx])
        im = imgs[idx]
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        if resize is not None:
            im = im.resize(resize)
        ax.imshow(im, cmap='gray')


# map class -> idx
label_to_idx = {
    'CHEETAH': 0,
    'OCELOT': 1,
    'CARACAL': 2,
    'LIONS': 3,
    'TIGER': 4,
    'PUMA': 5
}

idx_to_label = {
    0: 'CHEETAH',
    1: 'OCELOT',
    2: 'CARACAL',
    3: 'LIONS',
    4: 'TIGER',
    5: 'PUMA'
}


def make_dataset(imgs, labels, label_map, img_size):
    x = []
    y = []
    n_classes = len(list(label_map.keys()))
    for im, l in zip(imgs, labels):
        # preprocess img
        x_i = im.resize(img_size)
        x_i = np.asarray(x_i)

        # encode label
        y_i = label_map[l]

        x.append(x_i)
        y.append(y_i)
    return np.array(x).astype('float32'), np.array(y)


def save_model(model, filepath):
    """
    Save PyTorch model to a file.

    Args:
        model: PyTorch model to be saved.
        filepath (str): Path to save the model.
    """
    torch.save(model.state_dict(), filepath)


def load_model(model_class, filepath, device='cpu'):
    """
    Load PyTorch model from a file.

    Args:
        model_class: Model class (e.g., ConvNet) to instantiate.
        filepath (str): Path from which to load the model.
        device (str): Device to move the model to (default is 'cpu').

    Returns:
        model: Loaded PyTorch model.
    """
    model = model_class()  # change it to your own model class
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    return model


# Create new Dataset objects for the training and test datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


imgs, labels = load_imgs('./Dataset', ['CHEETAH', 'OCELOT', 'CARACAL', 'LIONS', 'TIGER', 'PUMA'])  # TODO CHANGE ME!!
X, y = make_dataset(imgs, labels, label_to_idx, (224, 224))
print('x shape: {}, y shape:{}'.format(X.shape, y.shape))
plot_sample(imgs, labels, 3, 3, resize=(224, 224))

################### Task 1 ###################
# general context setting
SEED = 20020309
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 32
EPOCHS = 1 + 20  # 100
LEARNING_RATE = 0.001
CONFIGS = {'seed': SEED,
           'device': device,
           'batch_size': BATCH_SIZE,
           'epochs': EPOCHS,
           'learning_rate': LEARNING_RATE}


def get_input():
    def safe_copy(*args):
        return map(lambda el: el.copy(), args)

    _x, _y = safe_copy(X, y)
    _x = _x / 255.0  # normalizing images
    _x = _x.reshape(_x.shape[0], -1)  # 1D vector image
    from sklearn.preprocessing import OneHotEncoder
    _y = OneHotEncoder().fit_transform(_y.reshape(-1, 1)).toarray()
    return x, y


def get_loads(_x, _y, _ratio, _permute=False, _transform=None):
    _x_train, _x_test, _y_train, _y_test = train_test_split(
        _x, _y, train_size=_ratio, random_state=CONFIGS['seed']
    )

    from torch.utils.data import TensorDataset, DataLoader
    # prepare data for training using TensorDataset and DataLoader
    _train_dataset = TensorDataset(_x_train, _y_train)
    _test_dataset = TensorDataset(_x_test, _y_test)

    # create dataLoader for both training and validation datasets
    batch_size = CONFIGS["batch_size"]
    _train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
    _test_loader = DataLoader(_test_dataset, batch_size=batch_size, shuffle=True)
    return _train_loader, _test_loader


x, y = get_input()
train_loader, test_loader = get_loads(x, y, 0.8)

# # prepare data for training
# X_tensor = torch.tensor(s, device=device)
# y_tensor = torch.tensor(y, dtype=torch.long, device=device)
# # one hot encode the labels (the y variable)
# y_one_hot = F.one_hot(y_tensor, num_classes=len(label_to_idx)).to(device)
#
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_tensor, y_one_hot, test_size=0.2, random_state=CONFIGS["seed"]
# )
#
# from torch.utils.data import TensorDataset, DataLoader
# # prepare data for training using TensorDataset and DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
#
# # create dataLoader for both training and validation datasets
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# print("Process dataset done!"
#     f"X_train.shape: {X_train.shape}"
#     f"X_val.shape: {X_val.shape}" # rename test
#     f"vectorized image size: {X_train.shape[1]}",
#     sep='\n')

from torch import nn
from torch.nn import functional as F


class FFNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(FFNN, self).__init__()
        # 3 layers
        self.fc1 = nn.Linear(in_features, 128)  # input layer with 128 hidden units
        self.fc2 = nn.Linear(128, 64)  # hidden layer with 64 hidden units
        self.fc3 = nn.Linear(64, num_classes)  # output layer with num_classes units

    def forward(self, x):
        # pass through layers with GELU activation
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)  # no activation for the output layer, output raw logits
        return x


def calculate_accuracy(y_true, y_pred):
    predicted_classes = torch.argmax(y_pred, dim=1)
    true_classes = torch.argmax(y_true, dim=1)
    correct = (predicted_classes == true_classes).sum().item()
    return correct


def load_to_device(*args):
    return map(lambda x: x.to(CONFIGS["device"]), args)


# plot training and test accuracies
def plot_train_test(train_acc, test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()


# init model, loss function criterion and optimizer
model = FFNN(x.shape[1], len(label_to_idx)).to(CONFIGS["device"])
criterion = nn.BCEWithLogitsLoss()  # better for classification with discrete values
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIGS["learning_rate"])

# train the model and get accuracies
train_acc = []
test_acc = []
best_val_accuracy = 0
for epoch in range(CONFIGS["epochs"]):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = load_to_device(inputs, labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # compute accuracy
        correct += calculate_accuracy(labels, outputs)
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_acc.append(train_accuracy)

    # validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = load_to_device(inputs, labels)
            outputs = model(inputs)
            val_correct += calculate_accuracy(labels, outputs)
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    test_acc.append(val_accuracy)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{CONFIGS['epochs']}]",
              f"Train Acc: {train_accuracy:.2f}",
              f"Val Acc: {val_accuracy:.2f}",
              sep=", ")

# used later on for statistical comparison
val_acc_t1 = test_acc.copy()
plot_train_test(train_acc, test_acc)
