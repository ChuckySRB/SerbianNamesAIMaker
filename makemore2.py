"""
Моја анализа резулта:

Модел који се користим је направљен већ 2003,
и резултати који су добијени су већински смислени и налик на именима
које имамо у нашем језику. Задивљујуће је да направити нешто у
оволико једноставнијој и разумљивијој структури и видити задовољавајуће
резултате са овом комплексношћу.

Оно што фали моделу је додатна комплексност д аразуме односе слова унутар
речи и делова који их сачињавају. Осим саме комплексости, треба
имплементирати и додатну логуку која би моделу дала слободу да буде
креативнији при прављењу имена.

Надам се да ћу разумети како се долази и до ових могућности у
наредним корацима!

"""




import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from format_data import output_file

NAMES_MALE = "./data/имена_српска_мушка.txt"
NAMES_FEMALE = "./data/имена_српска_женска.txt"

RESULTS_MALE = "./results/имена_српска_мушка_makemore_2003.txt"
RESULTS_FEMALE = "./results/имена_српска_женска_makemore_2003.txt"

NAMES_FILE = NAMES_MALE


# Function to read in all words from the file
def read_words(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    # Convert the first letter of each word to lowercase
    words = [word[0].lower() + word[1:] if word else word for word in words]
    return words


# Build vocabulary of characters and mappings to/from integers
def build_vocab(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


# Build dataset function
def build_dataset(words, block_size, stoi):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


# Function to initialize model parameters
def init_model_params(block_size, num_of_letter, embedding_dimensions = 10, neurons = 200):
    C = torch.randn((num_of_letter, embedding_dimensions))
    W1 = torch.randn((block_size * embedding_dimensions, neurons))
    b1 = torch.randn(neurons)
    W2 = torch.randn((neurons, num_of_letter))
    b2 = torch.randn(num_of_letter)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters, C, W1, b1, W2, b2


# Function to split the dataset into training, development, and test sets
def split_data(words, block_size, stoi):
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi=stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi=stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi=stoi)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte

def loss(X, Y, C, W1, b1, W2, b2):
    # calculate loss
    emb = C[X]  # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    loss = F.cross_entropy(logits, Y)
    return loss

# Function for the training loop
def train(Xtr, Ytr, parameters, C, W1, b1, W2, b2):
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10 ** lre
    stepi = []
    lossi = []

    for i in range(100000):
        ix = torch.randint(0, Xtr.shape[0], (32,))

        # Forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update parameters
        lr = 0.1 if i < 30000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # Track stats
        stepi.append(i)
        lossi.append(loss.log10().item())

        if i % 1000 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    plt.plot(stepi, lossi)
    plt.show()
    return C, W1, b1, W2, b2


# Visualization function
def visualize(C, itos):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()


def generate_names(block_size, C, W1, b1, W2, b2, itos, num_of_names, output_file="generated_names.txt"):
    # Sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)

    # Otvori fajl za upis generisanih imena
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(num_of_names):
            out = []
            context = [0] * block_size  # Initialize with all ...
            while True:
                emb = C[torch.tensor([context])]  # (1, block_size, d)
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                out.append(ix)

            # Rekonstruiši ime i promeni prvo slovo u veliko
            name = ''.join(itos[i] for i in out)
            name = name.capitalize()  # Pretvori prvo slovo u veliko

            # Upis u fajl
            f.write(name + '\n')

            # Opcioni ispis u konzoli
            print(name)


# Main function to run the entire process
def makemore_2003(block_size, num_of_names, input_file, output_file):

    # Step 1: Load and preprocess data
    words = read_words(input_file)
    stoi, itos = build_vocab(words)
    num_of_letter = len(stoi)
    # Step 2: Split data

    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_data(words, block_size, stoi)
    print(f"Training dataset shape: {Xtr.shape}, {Ytr.shape}")

    # Step 3: Initialize model parameters
    parameters, C, W1, b1, W2, b2 = init_model_params(block_size,  num_of_letter, embedding_dimensions = 10, neurons = 200)

    # Step 4: Train the model
    C, W1, b1, W2, b2 = train(Xtr, Ytr, parameters, C, W1, b1, W2, b2)
    train_loss, val_loss, test_loss = loss(Xtr, Ytr, C, W1, b1, W2, b2), loss(Xdev, Ydev, C, W1, b1, W2, b2), loss(Xte, Yte, C, W1, b1, W2, b2)

    print("Train loss: " + str(train_loss))
    print("Val loss: " + str(val_loss))
    print("Test loss: " + str(test_loss))


    # Step 5: Visualize the result
    visualize(C, itos)
    print ("Иде гас!")

    # Step 6:  Generate words
    generate_names(block_size, C, W1, b1, W2, b2, itos, num_of_names, output_file)

def chose_gender(gender):
    if gender == 0:
        return NAMES_MALE, RESULTS_MALE
    else:
        return NAMES_FEMALE, RESULTS_FEMALE

if __name__ == '__main__':
    block_size = 3
    num_of_names = 100

    # Направи мушка имена
    input_file, output_file = chose_gender(0)
    makemore_2003(block_size, num_of_names, input_file, output_file)

    # Направи женска имена
    input_file, output_file = chose_gender(1)
    makemore_2003(block_size, num_of_names, input_file, output_file)
