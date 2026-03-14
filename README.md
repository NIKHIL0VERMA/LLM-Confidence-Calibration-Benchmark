# LLM Confidence Calibration Benchmark

This project evaluates the confidence reliability and calibration behavior of open-source large language models (LLMs) across multiple reasoning tasks. The framework benchmarks LLM predictions on diverse datasets and analyzes how well model-reported confidence aligns with actual correctness.

## 🎯 Goal
To analyze whether modern open-source LLMs are well-calibrated, and how calibration varies across different task types such as reasoning, binary decision making, common sense and factual truthfulness.

## 🛠️ Key Features
- **Evaluation across multiple open-source LLMs**
- **Multi-dataset benchmarking:** GSM8K, BoolQ, TruthfulQA, CommonSenseQA
- **Confidence extraction** from model responses
- **Semantic answer evaluation** using sentence embeddings
- **Calibration analysis** using metrics such as Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score
- **Reliability diagrams** and confidence distribution visualization

## 📂 Project Structure

This repository is designed to be fully reproducible on Kaggle as well as locally while maintaining all source code and analysis in one organized place.

```shell
├── notebooks/
│   ├── 01_dataset_creation.ipynb     # Extracts and prepares reference datasets
│   ├── 02_model_evaluation.ipynb     # Runs LLM inference & confidence extraction
│   └── 03_analysis_visualization.ipynb # Computes calibration metrics & creates plots
├── data/                             # Processed Parquet/CSV files 
│   ├── Confidence_calibration_study_dataset.csv
│   └── Confidence_calibration_study_dataset.parquet
├── results/                          # LLM Inference & Confidence Extraction Results
├── outputs/                          # Output figures, reliability diagrams, and tables
├── src/                              # Reusable Python scripts for local replication
├── README.md
└── requirements.txt                  # Dependencies needed for local replication
```

## 📊 Datasets
- **Processed Datasets for Evaluation:** [LLM Calibration Benchmark - Processed Datasets](https://www.kaggle.com/datasets/nikhil2003verma/llm-calibration-benchmark-processed-datasets)

## 🚀 How to Run (Kaggle Workflow)

The entire pipeline is split into three Kaggle Notebooks to manage computational constraints (especially GPU time for LLM inference). 

### Step 1: Dataset Creation
1. Open [`notebooks/01_dataset_creation.ipynb`](notebooks/01_dataset_creation.ipynb) in Kaggle.
2. Run the notebook to fetch and format GSM8K, BoolQ, TruthfulQA, and CommonSenseQA.
3. The notebook will output `.parquet` and `.csv` files.

### Step 2: Model Evaluation & Confidence Extraction
1. Open [`notebooks/02_model_evaluation.ipynb`](notebooks/02_model_evaluation.ipynb) in Kaggle.
2. Enable a GPU accelerator (e.g., T4 x2).
3. Mount the Kaggle Dataset created in Step 1.
4. Run inference for your chosen open-source LLMs.

### Step 3: Analysis & Visualization
1. Open [`notebooks/03_analysis_visualization.ipynb`](notebooks/03_analysis_visualization.ipynb) in Kaggle.
2. Mount the dataset containing the LLM predictions (from Step 2).
3. Run the notebook to calculate ECE, MCE, Brier Scores, and generate reliability diagrams.

## 💻 How to Run Locally

You can also run the evaluation pipeline locally using the extracted Python scripts.

### Prerequisites

1. Clone the repository and navigate into it.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face API token. Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

### Running the Scripts

1. **Dataset Creation**: Run the dataset creation script to download and prepare the datasets. The outputs will be saved in the `data/` directory.
   ```bash
   python src/01_dataset_creation.py
   ```
2. **Model Evaluation**: Run the evaluation script. This will load models, verify predictions, and extract confidence scores. Output files and checkpoints will be saved in the `results/` folder.
   ```bash
   python src/02_model_evaluation.py
   ```

## 📚 Motivation
While LLMs can produce highly confident responses, their confidence does not always reflect true correctness. Understanding and evaluating calibration is critical for deploying LLMs in real-world decision-making systems.
