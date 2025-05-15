## **Protein Scructure Prediction w/ Computational Model**
With the rise in neural networks being used in predicting how proteins will fold, this model attempts to recreate top models such as AlphaFold with lower complexity. 

## Background
AlphaFold2 is a multicomponent artificial intelligence system that uses machine learning to predict a protein’s 3D structure based on its amino acid sequence. AlphaFold, which was created by DeepMind, is trained on known protein structures and their amino acid sequences from the Protein Data Bank. It uses these known sequences to find relationships and predict how a new protein will fold. AlphaFold also incorporates attention mechanisms to model spatial relationships and refine predictions. Other models, such as RoseTTAFold, use similar techniques to simulate folding. These tools have been very impactful in the bioinformatics field by providing accurate predictions, which aids in drug discovery and disease research. 

![image](https://github.com/user-attachments/assets/5947bf02-c77c-46f8-8ba6-3d08625ea56e)

For my research project, I aim to recreate AlphaFold by creating a model that predicts how a protein will fold based on its amino acid structure. Along with this, I will do a deep dive into how AlphaFold works and create a presentation on its functionality. I am drawn to this project because it is something that I did in the past and hope to finish with this project. I am interested in machine learning, and this project is the perfect intersection between my interests and the content of the class. I hope to gain a greater understanding in how neural networks function and learn more about bioinformatics. 

## Alphafold's Functionality
AlphaFold’s system works in several stages:
1. Input Representation: Converts amino acid sequences into numerical features, often supplemented by multiple sequence alignments (MSAs).
2. Pairwise Attention: Captures relationships between residues, building a spatial map of probable distances and angles.
3. Structure Module: Predicts 3D atomic coordinates based on learned spatial maps and sequence embeddings.
4. Iterative Refinement: Recycles predictions multiple times to correct initial errors and improve final outputs.

![image](https://github.com/user-attachments/assets/27f5f46c-d66a-44c8-a2b0-d40f0edf04f9)

Although AlphaFold2 is very complex, my project focuses on capturing the basic idea:
1. Starting from sequence information.
2. Using attention mechanisms to model residue interactions.
3. Predicting a 3D Cα trace.

## Development
### 1. Data Retrieval
- Retrieve the first 300 PDB IDs that have the tag "Homo Sapiens"
- IDs that return an error are skipped over, most likely due to incompatibility or returning a blank value for the sequence

### 2. Preprocessing
- **FASTA Sequence Preprocessing**:  
  - Converts an amino acid sequence into a one-hot encoded tensor (20 amino acids → 20-dimensional vectors).
- **PDB File Retrieval**:  
  - Downloads the corresponding protein structure (.pdb file) from the RCSB Protein Data Bank.
- **Cα Coordinate Extraction**:  
  - Parses the PDB file and extracts the x, y, z coordinates of **Cα atoms** (representing the protein backbone).
I chose to focus on alpha-carbon atoms due to the nature of the model. Since the model is a scaled down version lacking a lot of the things AlphaFold has, I chose to only focus on the protein backbone. Alpha-carbons are the central point in the backbone of every amino acid, so I only extracted their positions. 

### 3. Model Architecture
- **Input**:  
  - Variable-length one-hot encoded sequence.
  - Each sequences has a different length, so the program needs to adapt
- **Layers**:
  - Dense layer with 128 units and GELU activation
    - Projects the one-hot encoded sequence into a higher-dimensional feature space.
    - GELU activation provides a smoother, more natural gradient flow than ReLU, helping the model learn complex patterns.
  - Layer Normalization layer
    - Normalizes features across each input to stabilize and speed up training.
  - Multi-Head Attention layer (8 heads, key dimension = 32)
    - Let's the model learn different patterns in parallel, allowing it to learn more complex relationships. 
  - Layer Normalization layer
    - Normalizes the outputs
  - Dense layer with 128 units and GELU activation
  - Final Dense layer that predicts 3D coordinates (x, y, z) for each amino acid residue

### 4. Training
- **Loss Function**: Mean Squared Error (MSE).
- **Training Setup**:
  - Trained on multiple sequence-structure pairs. 
  - 200 epochs of training to fit predicted Cα coordinates to true coordinates.

### 5. Evaluation
- **Metric**:  
  - Calculates **Root Mean Square Deviation (RMSD)** between the predicted and true Cα trace.
  - RMSD is commonly used in protein structure prediction, as it measures the difference in the prediction and the actual based on Euclidean distance. 
- **Visualization**:  
  - Generates a 3D plot comparing the predicted trace against the actual structure, allowing visual inspection of prediction accuracy.

## Results
With a 2:1 train/validation setup, I got an average RMSD score of 26.29953406627689 Å

Below is a graph of one of the better generations, with a score of 9.252 Å
![image](https://github.com/user-attachments/assets/81c9e9b5-bd45-4aa3-b037-096eb4920eac)

Obviously, the results are not ideal. This is to be expected. 

## Future Steps
There are several areas for improvement in future work. The results with the given work have a lot of room for improvement. While the general structure is there, a lot needs to be added on to properly imitate AlphaFold and accurately predict proteins structures. 

Expand Training Data:
  - Increase the size and diversity of the dataset by including more PDB entries from a variety of organisms, not just Homo sapiens.
  - Include longer and more structurally complex proteins to better challenge and train the model.

Model Improvements:
- Implement a recycling mechanism, where the model's predictions are fed back into itself to refine outputs, mimicking AlphaFold’s iterative approach.
- Explore transformer-based architectures beyond simple multi-head attention for more powerful spatial learning.

Predict More than Cα:
- Expand the model to predict side chain atoms or full-atom coordinates, instead of just the alpha-carbon trace, for more realistic structure predictions.

Advanced Loss Functions:
- Experiment with additional loss functions, such as distance matrix loss, to guide the model towards better global folding.

Benchmarking:
- Compare the model’s performance against standard datasets used in protein structure prediction (e.g., CASP datasets).
