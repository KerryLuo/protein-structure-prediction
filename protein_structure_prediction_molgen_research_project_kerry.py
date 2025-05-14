import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import requests

def preprocess_fasta(fasta):
    amino_acids = [fasta[i:i+1] for i in range(0, len(fasta))]
    STRINGS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    LABELS = [i for i in range(len(STRINGS))]

    def get_label_from_string(string, dataframe):
        result = dataframe.loc[dataframe['string'] == string, 'label']
        if not result.empty:
            return result.iloc[0]
        else:
            return None

    df = pd.DataFrame({'string': STRINGS, 'label': LABELS})
    for i in range(len(amino_acids)):
        amino_acids[i] = get_label_from_string(amino_acids[i], df)

    return tf.one_hot(amino_acids, 20)

def create_minifold_model():
    inputs = layers.Input(shape=(None, 20))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(3)(x)
    return models.Model(inputs, outputs)

def fetch_pdb_content(pdb_id):
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch PDB file {pdb_id}: {e}")
        return None

def pdb_to_arr(pdb_text):
    pdb_lines = pdb_text.strip().split('\n')
    return [line.split() for line in pdb_lines if line.startswith("ATOM")]

def extract_ca_coordinates(pdb_2d):
    coords = []
    for line in pdb_2d:
        if line[2] == "CA":
            coords.append([float(line[6]), float(line[7]), float(line[8])])
    return coords

def calculate_rmsd(pred, true):
    pred, true = np.array(pred), np.array(true)
    min_len = min(len(pred), len(true))
    pred, true = pred[:min_len], true[:min_len]
    return np.sqrt(np.mean((pred - true)**2))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_comparison(pred, true):
    pred, true = np.array(pred), np.array(true)
    min_len = min(len(pred), len(true))
    pred, true = pred[:min_len], true[:min_len]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*pred.T, label='Predicted', color='blue')
    ax.plot(*true.T, label='True', color='red')
    ax.legend()
    ax.set_title("Predicted vs True Cα Trace")
    plt.show()

###############################################################

# Example
fasta = "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"

# Example matching PDB (Cytochrome C)
pdb_id = "3ZOW"
pdb_txt = fetch_pdb_content(pdb_id)
pdb_arr = pdb_to_arr(pdb_txt)
ca_coords = extract_ca_coordinates(pdb_arr)

X = preprocess_fasta(fasta).numpy()
y = np.array(ca_coords)

min_len = min(X.shape[0], y.shape[0])
X, y = X[:min_len], y[:min_len]
X = X[None, ...]
y = y[None, ...]

model = create_minifold_model()
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=2)

X_test = tf.expand_dims(preprocess_fasta(fasta), axis=0)

predicted_coords = model.predict(X_test)[0]

rmsd = calculate_rmsd(predicted_coords, ca_coords)
print(f"RMSD between prediction and true structure: {rmsd:.3f} Å")

plot_comparison(predicted_coords, ca_coords)
