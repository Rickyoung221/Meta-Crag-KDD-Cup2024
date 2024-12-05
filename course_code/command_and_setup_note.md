# Project Setup and Usage Guide

## Project Setup

### 1. Set Up the VM

- Configure and initialize a virtual machine (VM) for the project.

### 2. Add SSH Public Key to VM

- Add your SSH public key to the VM for secure remote access.

### 3. Connect to the Remote Host from VS Code

- Use VS Code’s remote SSH extension to connect to the VM.

### 4. Hugging Face Login and Environment Setup

- Log in to your Hugging Face account.
- Set up the required environment for the project.

------

## Running the Baseline Code

### Start the VLLM Server

Run the following command to start the VLLM server:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --gpu_memory_utilization=0.95 \
    --tensor_parallel_size=1 \
    --dtype="half" \
    --port=8088 \
    --enforce_eager
```

### Offline Inference Example

Check for the following file in `/var/tmp`:

```
5c67febead924a30c0ad73283fca422fade2ccd58797fe886109a313b4f719c2meta-llama-Llama-3.2-1B-Instruct.lock
```

If it exists, delete the lock file.

Run offline inference with:

```bash
python generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name "rag_baseline" \
    --llm_name "meta-llama/Llama-3.2-1B-Instruct"
```

------

## Running Python Code

### Install Required Libraries

Install `lxml`:

```bash
pip install lxml
```

### Hugging Face CLI Login

Log in to Hugging Face with your token:

```bash
huggingface-cli login --token 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
```

### Run the Generate Script

1. Navigate to the 

   ```
   course_code
   ```

    directory:

   ```bash
   cd course_code
   ```

2. Set the GPU environment:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

3. Execute the script:

   ```bash
   python generate.py \
       --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
       --split 1 \
       --model_name "new_rag_baseline" \
       --llm_name "meta-llama/Llama-3.2-1B-Instruct"
   ```

### Evaluate the Model

Run the evaluation script:

```bash
python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name "new_rag_baseline" \
    --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
    --max_retries 10
```

------

## Monitoring System Usage

Monitor CPU and GPU usage with the following commands in separate terminals:

1. GPU usage:

   ```bash
   watch -n 0.5 nvidia-smi
   ```

2. CPU usage:

   ```bash
   watch -n 0.5 "top -b -n1 | head -n 15"
   ```

------

## Task 2: CRAG API Wrapper

The `utils/cragapi_wrapper.py` file defines the `CRAG` class, which facilitates interaction with the CRAG server. This server provides APIs across various domains such as Open Domain, Movies, Finance, Music, and Sports. Each method corresponds to a specific API endpoint and returns a JSON-formatted response.

### Overview of the CRAG Class and Methods

#### General Method Format

Each method sends a `POST` request to the CRAG server and retrieves a JSON response.

#### Supported Domains and Methods

1. **Open Domain**
   - `open_search_entity_by_name(query: str) -> dict`: Search for an entity by name.
   - `open_get_entity(entity: str) -> dict`: Retrieve detailed information about an entity.
2. **Movies**
   - `movie_get_person_info(person_name: str) -> dict`: Fetch information about a person in the film industry.
   - `movie_get_movie_info(movie_name: str) -> dict`: Get details about a specific movie.
   - `movie_get_year_info(year: str) -> dict`: Retrieve movies released in a specific year.
   - `movie_get_movie_info_by_id(movie_id: int) -> dict`: Get movie details by ID.
   - `movie_get_person_info_by_id(person_id: int) -> dict`: Get person details by ID.
3. **Finance**
   - `finance_get_company_name(query: str) -> dict`: Search for a company by name.
   - `finance_get_ticker_by_name(query: str) -> dict`: Get the ticker symbol for a company.
   - `finance_get_price_history(ticker_name: str) -> dict`: Retrieve stock price history.
   - `finance_get_detailed_price_history(ticker_name: str) -> dict`: Retrieve detailed stock price history.
   - `finance_get_dividends_history(ticker_name: str) -> dict`: Get dividend history for a stock.
   - `finance_get_market_capitalization(ticker_name: str) -> dict`: Retrieve market capitalization.
   - `finance_get_eps(ticker_name: str) -> dict`: Retrieve earnings per share (EPS).
   - `finance_get_pe_ratio(ticker_name: str) -> dict`: Retrieve the price-to-earnings (PE) ratio.
   - `finance_get_info(ticker_name: str) -> dict`: Get comprehensive financial information about a company.
4. **Music**
   - `music_search_artist_entity_by_name(artist_name: str) -> dict`: Search for an artist by name.
   - `music_search_song_entity_by_name(song_name: str) -> dict`: Search for a song by name.
   - `music_get_billboard_rank_date(rank: int, date: str = None) -> dict`: Retrieve Billboard rankings for a specific date.
   - `music_get_artist_all_works(artist_name: str) -> dict`: Fetch all works by an artist.
   - Additional methods include retrieving Grammy award details, artist and song information, release dates, and more.
5. **Sports**
   - `sports_soccer_get_games_on_date(team_name: str, date: str) -> dict`: Retrieve soccer games for a specific date.
   - `sports_nba_get_games_on_date(team_name: str, date: str) -> dict`: Retrieve NBA games for a specific date.
   - `sports_nba_get_play_by_play_data_by_game_ids(game_ids: List[str]) -> dict`: Get play-by-play data for NBA games by their IDs.

------

### Example Usage

Here is an example of how to use the `CRAG` class:

```python
# Create a CRAG client instance
crag_client = CRAG()

# Open Domain Example
entity_search_result = crag_client.open_search_entity_by_name("example entity")
print(entity_search_result)

# Movie Example
movie_info = crag_client.movie_get_movie_info("Inception")
print(movie_info)

# Finance Example
price_history = crag_client.finance_get_price_history("AAPL")
print(price_history)

# Sports Example
soccer_games = crag_client.sports_soccer_get_games_on_date("2023-12-01", "FC Barcelona")
print(soccer_games)
```

By using the above methods, you can seamlessly interact with the CRAG server to retrieve data across multiple domains.