# !pip install rcsbsearchapi

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import requests
from os import write
from rcsbsearchapi import AttributeQuery
import requests
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def fetch_sequence(pdb_id):
    pdb_id = pdb_id.upper()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        fasta_text = response.text

        lines = fasta_text.strip().split('\n')
        sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
        return sequence

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {pdb_id}: {e}")
        return None


###################################################################################

def create_minifold_model():
    inputs = layers.Input(shape=(None, 20))

    # Project one-hot sequence into higher-dimensional space
    x = layers.Dense(128, activation='gelu')(inputs)  # GELU is a smoother gradient than RELU (better for complex patterns) but is more computationally expensive. Since the model is pretty small, this doesn't have too much or a downside, which is why i chose to include this
    x = layers.LayerNormalization()(x)

    # Multi-head attention
    attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)  # more heads
    x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation='gelu')(x)

    outputs = layers.Dense(3)(x)  # Predict (x, y, z)

    return models.Model(inputs, outputs)

###################################################################################

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

###################################################################################

# repurposed code from j-term
# Construct a query searching for structures from humans
query = AttributeQuery(
    attribute="rcsb_entity_source_organism.scientific_name",
    operator="exact_match",
    value="Homo sapiens"
)

test_split = 200
sequences = list(query())[0:test_split]
print(sequences[0:10])

train_data = []

for pdb_id in tqdm(sequences):
    try:
        fasta_sequence = fetch_sequence(pdb_id)
        pdb_text = fetch_pdb_content(pdb_id)

        if fasta_sequence is None or pdb_text is None:
            print(f"Skipping {pdb_id}: missing sequence or PDB.")
            continue

        x = preprocess_fasta(fasta_sequence).numpy()
        pdb_arr = pdb_to_arr(pdb_text)
        y = np.array(extract_ca_coordinates(pdb_arr))

        min_len = min(x.shape[0], y.shape[0])
        x, y = x[:min_len], y[:min_len]

        train_data.append((x, y))

    except Exception as e:
        print(f"Skipping {pdb_id}: error during processing. ({e})")
        continue

inputs = pad_sequences([x for x, y in train_data], padding="post", dtype="float32")
targets = pad_sequences([y for x, y in train_data], padding="post", dtype="float32")

model = create_minifold_model()
model.compile(optimizer='adam', loss='mse')
model.fit(inputs, targets, epochs=200)

###################################################################################

validation = list(query())[test_split:test_split + 100]

RMSD_total = 0
RMSD_count = 0

for test_pdb_id in validation:
  try:
    fasta_sequence = fetch_sequence(test_pdb_id)
    pdb_text = fetch_pdb_content(test_pdb_id)

    X_test = tf.expand_dims(preprocess_fasta(fasta_sequence), axis=0)
    pdb_arr = pdb_to_arr(pdb_text)
    ca_coords = np.array(extract_ca_coordinates(pdb_arr))

    predicted_coords = model.predict(X_test)[0]

    rmsd = calculate_rmsd(predicted_coords, ca_coords)

    print(f"RMSD between prediction and true structure for {test_pdb_id}: {rmsd:.3f} Å")
    plot_comparison(predicted_coords, ca_coords)

    RMSD_total += rmsd
    RMSD_count += 1
  except:
    pass

print("Average RMSD:", RMSD_total/RMSD_count)
# print(f"RMSD between prediction and true structure for {test_pdb_id}: {rmsd:.3f} Å")

# plot_comparison(predicted_coords, ca_coords)
