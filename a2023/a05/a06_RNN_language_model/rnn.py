
# Prepare the dataset
# -------------------

import numpy as np

# Load and preprocess the text
with open("input_small.txt", "r") as f:
  text = f.read().lower()
  text = ''.join(c for c in text if c.isalnum() or c.isspace())

# Create dictionaries
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Convert text to integers
int_text = [char_to_int[c] for c in text]

# Create input and target sequences
sequence_length = 50
X, y = [], []

for i in range(len(int_text) - sequence_length):
  X.append(int_text[i:i + sequence_length])
  y.append(int_text[i + 1:i + sequence_length + 1])

X = np.array(X)
y = np.array(y)


# Implement a simple RNN model in PyTorch
# ---------------------------------------

import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
  def __init__(self, input_size, embed_size, hidden_size, output_size):
    super(SimpleRNN, self).__init__()
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.embedding = nn.Embedding(input_size, embed_size)
    self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden):
    x = self.embedding(x)
    x, hidden = self.rnn(x, hidden)
    x = self.fc(x)
    return x, hidden

  def init_hidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size)

input_size = len(chars)
embed_size = 128
hidden_size = 256
output_size = len(chars)

model = SimpleRNN(input_size, embed_size, hidden_size, output_size)


# Define the loss function and the optimizer
# -----------------------------------------

import torch.optim as optim

# Set the learning rate
lr = 0.001

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


# Train the model
# ---------------

num_epochs = 5

# Set the batch size
batch_size = 64

# Train the model
for epoch in range(num_epochs):
  # Loop over the input-target pairs in the dataset
  for i in range(0, len(X), batch_size):
    # Get the actual batch size for the current iteration
    actual_batch_size = min(batch_size, len(X) - i)

    # Reset the hidden state
    hidden = model.init_hidden(actual_batch_size)

    # Detach the hidden state from its history
    hidden.detach_()

    # Get a batch of input and target sequences
    input_batch = torch.tensor(X[i:i+actual_batch_size], dtype=torch.long)
    target_batch = torch.tensor(y[i:i+actual_batch_size], dtype=torch.long)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass: pass the input and hidden state to the model
    output, hidden = model(input_batch, hidden)

    # Reshape the output and target_batch
    output = output.view(-1, output.shape[2])
    target_batch = target_batch.view(-1)

    # Compute the loss
    loss = criterion(output, target_batch)

    # Backward pass: compute the gradients
    loss.backward()

    # Update the model parameters
    optimizer.step()

  # Print the loss for this epoch
  print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# Function to generate new text
# --------------------------------

def generate_text(model, initial_sequence, n_chars, temperature=1.0):
  model.eval()  # Set the model to evaluation mode
  initial_sequence = initial_sequence.lower()  # Convert the initial sequence to lowercase

  # Only keep characters that are present in char_to_int
  initial_sequence = ''.join(c for c in initial_sequence if c in char_to_int)
  generated_sequence = initial_sequence

  # Convert the initial sequence to a tensor
  input_sequence = torch.tensor([char_to_int[c] for c in initial_sequence], dtype=torch.long).unsqueeze(1)

  # Initialize the hidden state
  hidden = model.init_hidden(input_sequence.size(1))

  # Generate new characters
  for _ in range(n_chars):
    # Reshape input_sequence to have a sequence length of 1
    input_sequence = input_sequence.view(1, -1)
    output, hidden = model(input_sequence, hidden)
    hidden = hidden.detach()  # Detach the hidden state

    # Apply the temperature and sample the output probabilities
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
    print(top_i.item())

    # Add the generated character to the sequence
    generated_char = int_to_char[top_i.item()]
    generated_sequence += generated_char

    # Update the input sequence
    input_sequence = torch.tensor([char_to_int[generated_char]], dtype=torch.long).unsqueeze(1)

  return generated_sequence


# Generate new text
# --------------------------------

initial_sequence = "To be, or not to be"
n_chars = 200
temperature = 1.0

generated_text = generate_text(model, initial_sequence, n_chars, temperature)
print(generated_text)



