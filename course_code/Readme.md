# CS245 Final Project

This repository contains our implementation for the Meta KDD Cup 2024 CRAG Benchmark tasks. The project focuses on a Retrieval-Augmented Generation (RAG) system for accurate, hallucination-free answers across three tasks:

- **Task 1:** Retrieval Summarization
- **Task 2:** Knowledge Graph and Web Retrieval
- **Task 3:** Advanced Synthesis and Reasoning

---

## Project Structure

Platform used: Vast AI, A-100.

- **`new_rag_baseline/`**: Codebase for Task 1 with retrieval and summarization improvements.
- **`task2_rag_baseline/`**: Extension of Task 1, integrating Knowledge Graphs.
- **`task3_advanced_baseline/`**: Advanced reasoning and synthesis across structured and unstructured data.
- **`data/`**: Contains datasets for all tasks.
- **`output/`**: Stores generated predictions and evaluation results.
- **`docs/`**: Documentation, including dataset details and methodology.

---

## Tasks Overview

### Task 1: Retrieval Summarization

Summarize retrieved content into concise, accurate answers while avoiding hallucinations.

- **Model Used**: `meta-llama/Llama-3.2-3B-Instruct`.

- Performance Summary

  :

  | Metric             | Vanilla Baseline | RAG Baseline | New RAG Baseline |
  | ------------------ | ---------------- | ------------ | ---------------- |
  | Accuracy           | 12.96%           | 22.92%       | 39.1%            |
  | Exact Accuracy     | 0.37%            | 3.37%        | 4.94%            |
  | Hallucination Rate | 14.01%           | 28.31%       | 45.2%            |

For task 1, switch to branch "both" to see detailed implementation and evaluation result in output folder.

For task 2, switch to branch "task2" to see detailed implementation.

### Task 2: Knowledge Graph and Web Retrieval

Enhances Task 1 by incorporating structured data (Knowledge Graphs) using mock APIs for improved accuracy.

### Task 3: Advanced Synthesis and Reasoning

Combines multiple data sources, including structured data (Knowledge Graphs) and unstructured web data, to enable advanced synthesis and reasoning for complex queries.

- **Objective**: Address complex, multi-step questions requiring reasoning over multiple data modalities.

---

## Setup Instructions

### Prerequisites

- Python 3.10
- CUDA-compatible GPU for optimal performance
- Hugging Face and vLLM

### Environment Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/Rickyoung221/UCLA-CS245-FALL2024.git
   cd UCLA-245-FALL2024
   ```

2. Create a virtual environment:

   ```bash
   conda create -n crag python=3.10
   conda activate crag
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure Hugging Face:

   ```bash
   huggingface-cli login --token "YOUR_ACCESS_TOKEN"
   export CUDA_VISIBLE_DEVICES=0
   ```

5. Runnign the vllm server:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager
   ```

---

## Running the Tasks

### Task 1: Retrieval Summarization

1. Download the dataset: [Task 1 Data](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files), or git clone from *https://huggingface.co/datasets/Rickyoung0221/crag/tree/main* (recommend) save in the `./data`.

2. Run the script for inference:

   ```bash
   cd course_code

   python generate.py \
       --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
       --split 1 \
       --model_name "new_rag_baseline" \
       --llm_name "meta-llama/Llama-3.2-1B-Instruct"
   ```

3. Results are saved in `output/task1/`.

4. Evaluate

```bash
python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name "new_rag_baseline" \
    --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
    --max_retries 10
```

---

### Task 2: Knowledge Graph and Web Retrieval

1. Download the dataset and mock APIs: [Task 2 Data](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/knowledge-graph-and-web-retrieval/dataset_files)

   1. Generate an SSH key in your gpu server:

      ```bash
      ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
      cat ~/.ssh/id_rsa.pub
      ```

   2. Copy the generated public key and add it to the [GitLab SSH Key Settings Page](https://gitlab.aicrowd.com/-/user_settings/ssh_keys/10939).

      Before cloning the repository, ensure that Git Large File Storage (LFS) is installed:

      - Install Git LFS:

      ```bash
      brew install git-lfs
      ```

      - Initialize Git LFS:

      ```bash
      git lfs install
      ```

   3. Clone the Mock-APi Repository

      Use SSH to clone the CRAG-Mock-API repository:

   ```bash
   git clone git@gitlab.aicrowd.com:aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api.git
   ```

   4. Set Up Environment and Start the API Server

      - Navigate to the cloned repository directory:

        ```bash
        cd crag-mock-api
        ```

      - Install dependencies and start the API server (refer to the repository documentation for detailed commands).

        ```bash
        pip install -r requirements.txt
        ```

      - Start the server

        ```bash
        uvicorn server:app --reload
        ```

        Then visit `http://127.0.0.1:8000/docs` to check the APi documentation and test.

2. Run the script for inference:

   ```bash
   python generate.py \
   --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
   --split 1 \
   --model_name "task2_rag_baseline" \
   --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
   ```

3. Evaluate:

   ```bash
   python evaluate.py \
     --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
     --model_name "task2_rag_baseline" \
     --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
     --max_retries 20
   ```

---

### Task 3: Advanced Synthesis and Reasoning

1. Download the dataset: [Task 3 Data](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/advanced-synthesis-and-reasoning/dataset_files) or clone from *https://huggingface.co/datasets/Rickyoung0221/crag/tree/main* (recommend, much faster to your gpu server)

2. Run the script for inference:

   ```bash
   python generate.py \
   --dataset_path "data/crag_task_3_dev.jsonl.bz2" \
   --split 1 \
   --model_name "task3_advanced_baseline" \
   --llm_name "meta-llama/Llama-3.2-1B-Instruct"
   ```

3. Results are saved in `output/task3/`.

4. Evaluate: （remember extract the dataset file)

   ```bash
   python evaluate.py \
   --dataset_path "data/crag_task_3_dev_v4.jsonl.bz2" \
   --model_name "task3_baseline" \
   --llm_name "meta-llama/Llama-3.2-1B-Instruct"
   ```

---

## Additional Resources

- [Reference Github Repo](https://github.com/USTCAGI/CRAG-in-KDD-Cup2024/tree/master)
- [Official Competition Site](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
