# Pathogen Prediction from Metagenomic Data

Welcome to the repository for our advanced machine learning model designed to predict pathogen presence in metagenomic data.
This tool leverages cutting-edge deep learning techniques to classify and identify potential pathogenic agents,
aiding researchers and healthcare professionals in rapid and accurate pathogen detection.

## Features

- **Accurate Pathogen Classification**: Identifies bacterial, eukaryotic, and viral pathogens with high precision.
- **Comprehensive Dataset Support**: Trained on extensive datasets to ensure robustness and reliability.
- **Scalable Architecture**: Optimized for both small-scale and large-scale metagenomic analyses.
- **GPU Acceleration**: Utilizes GPU for faster processing and analysis.
- **User-Friendly Interface**: Easy-to-use scripts for data preparation, model training, and prediction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nicolaedrabcinski/metapathpredict.git
   cd metapathpredict
   ```
   
2. Activate Conda:
   ```bash
   conda activate
   ```

3. Install Mamba:
   ```bash
   conda install mamba -y
   ```

4. Import dependencies from the environment file:
   ```bash
   mamba env create -f envs/environment.yml
   ```

5. Activate the new environment:
   ```
   conda activate metapathpredict
   ```

## Usage

There are three main parts of the project: preparing the dataset, training models, and obtaining predictions. These parts are located in the workflow folder. You don't have to run them manually. Everything is configured in the bash script named scripts/run.sh.

## Configuration

Please note that everything in the project is configured for a specifig VM.
You need to modify only 'main_config.json' to define correct paths for machine.

## Running

1. Make the script executable:
   ```bash
   chmod +x scripts/run.sh
   ```

2. Run!
   ```bash
   bash scripts/run.sh
   ```

## Contributing

We welcome contributions from the community. Please submit pull requests or open issues for any bugs or feature requests.

## Note

This group project was developed during the EEBG Summer School 2024 and would not have been possible without the valuable contributions of the following participants: Karyna Kystsiv, Alla Kovalchuk, Maxim Comarov, Martiniuc Anghelina, Anna Opanasenko, Victoria Loziak, Eugeniu Cotlabuga, Volodymyr Khailov, Erik Savchyn.
