TODO:
1. add in-depth description of everything the code does
2. add more training data
3. test on different protein (not cytochrome c), more like something we did in class like a crispr protein


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
### 1. Preprocessing
- **FASTA Sequence Preprocessing**:  
  - Converts an amino acid sequence into a one-hot encoded tensor (20 amino acids → 20-dimensional vectors).
- **PDB File Retrieval**:  
  - Downloads the corresponding protein structure (.pdb file) from the RCSB Protein Data Bank.
- **Cα Coordinate Extraction**:  
  - Parses the PDB file and extracts the x, y, z coordinates of **Cα atoms** (representing the protein backbone).

### 2. Model Architecture
- **Input**:  
  - Variable-length one-hot encoded sequence.
  - This was 
- **Layers**:
  - Dense layer with 64 units and ReLU activation.
  - Layer Normalization layer.
  - Multi-Head Attention layer (4 heads, key dimension = 16).
  - Another Layer Normalization layer.
  - Dense layer with 128 units and ReLU activation.
  - Final Dense layer that predicts 3D coordinates (x, y, z) for each amino acid residue.

This architecture captures basic spatial relationships between amino acids using attention, similar in spirit to AlphaFold's mechanism.

### 3. Training
- **Loss Function**: Mean Squared Error (MSE).
- **Training Setup**:
  - Trained on multiple sequence-structure pairs. 
  - 200 epochs of training to fit predicted Cα coordinates to true coordinates.

### 4. Evaluation
- **Metric**:  
  - Calculates **Root Mean Square Deviation (RMSD)** between the predicted and true Cα trace.
- **Visualization**:  
  - Generates a 3D plot comparing the predicted trace against the actual structure, allowing visual inspection of prediction accuracy.

## Results
### Minimal Training (for demonstration)
In the first iteration, for demonstration purposes, the model was trained on just one sequence-structure pair. This resulted in a RMSD value of 11.209 Å. This was unoptimal, as an optimal RMSD value is below 3 Å.
![image](https://github.com/user-attachments/assets/5585af08-42c3-4e2a-8a61-63525d9b529a)

### With Training



## Future Steps


## Resources
