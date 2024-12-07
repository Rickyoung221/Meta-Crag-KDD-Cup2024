import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import vllm
import requests
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

#reference: https://github.com/USTCAGI/CRAG-in-KDD-Cup2024/tree/master
#reference: https://blog.csdn.net/m0_59164520/article/details/143694213
#dataset: https://huggingface.co/datasets/Rickyoung0221/crag/tree/main

#### CONFIG PARAMETERS ---
# Configuration parameters for chunking, context sizes, and model inference
NUM_CHUNK_PER_SENTENCE = 3             # Number of sentences combined into one chunk
NUM_CONTEXT_SENTENCES = 20             # Number of top relevant chunks (sentences) to use as context
MAX_CONTEXT_SENTENCE_LENGTH = 1000     # Max length for an individual context sentence
MAX_CONTEXT_REFERENCES_LENGTH = 4000   # Max total length of references (web + KG) to pass to the model
AICROWD_SUBMISSION_BATCH_SIZE = 1      # Batch size for inference during evaluation
VLLM_TENSOR_PARALLEL_SIZE = 1          # Parallelism setting for vLLM inference
VLLM_GPU_MEMORY_UTILIZATION = 0.85     # GPU memory utilization ratio for vLLM
SENTENCE_TRANSFORMER_BATCH_SIZE = 32   # Batch size for embedding calculation in SentenceTransformer
#### CONFIG PARAMETERS END---
class EntityExtractor:
    """Extracts entities such as time, company names, and person names from a query string.
    This is used for downstream tasks such as querying a Knowledge Graph (KG) with more specific constraints.
    """

    @staticmethod
    def extract_time(query: str) -> Optional[str]:
        """Extract time-related information (e.g., year or date) from a query string.
        
        The method checks for various patterns, including years like 1990, 2023,
        and dates formatted as 'YYYY-MM-DD' or 'Month Day, Year'.
        
        Returns:
            The first matched year or date string if found, otherwise None.
        """
        # Patterns to identify possible years
        year_patterns = [
            r'\b(?:19|20)\d{2}\b',  # Matches years 1900-2099 (e.g., 1999, 2021)
            r'\'\d{2}\b',           # Matches short form years like '99 or '23
            r'\b\d{4}\b'            # Matches any 4-digit number which might be a year
        ]
        
        # Try to match year patterns
        for pattern in year_patterns:
            years = re.findall(pattern, query)
            if years:
                year = years[0]
                # Convert short year formats like '23 to full format 2023 or 1923
                if '\'' in year:
                    year = '20' + year[1:] if int(year[1:]) < 50 else '19' + year[1:]
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                return year

        # Patterns to identify full date formats like '2021-09-01' or 'Sept 1, 2021'
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        ]
        # Try to match date patterns
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            if dates:
                return dates[0]
        
        return None

    @staticmethod
    def extract_company_name(query: str) -> Optional[str]:
        """Extract a potential company name from the query text.
        
        We look for patterns that match typical company naming conventions including
        suffixes like Inc, Corp, LLC, etc. Also tries to handle uppercase acronyms.
        
        Returns:
            A matched company name string if found, otherwise None.
        """
        # Common company suffix patterns
        suffixes = r'\b(?:Inc|Corp|Ltd|LLC|Company|Co|Group|Holdings|Technology|Tech|Industries|International)\b'
        prefixes = r'\b(?:The)\b'
        
        company_patterns = [
            rf'(?:{prefixes}\s+)?[A-Z][A-Za-z0-9\s&\'-]+(?:\s+{suffixes})?',  # e.g., "The Acme Inc"
            rf'[A-Z][A-Za-z0-9\s&\'-]+(?:\s+{suffixes})',                    # must end with a known suffix
            r'[A-Z]{2,}(?:\s+[A-Z]{2,})*'                                     # acronyms like IBM, MSFT
        ]
        
        for pattern in company_patterns:
            companies = re.findall(pattern, query)
            if companies:
                # If multiple matches, return the longest to get the most descriptive name
                return max(companies, key=len).strip()
        return None

    @staticmethod
    def extract_person_name(query: str) -> Optional[str]:
        """Extract a potential person's name from the query text.
        
        The pattern tries to match typical Western name formats, including:
        Firstname Lastname, potentially with 'van', 'von', etc., and middle initials.
        
        Returns:
            A matched person name string if found, otherwise None.
        """
        name_patterns = [
            r'[A-Z][a-z]+(?:\s+(?:van|von|de|la|le))?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'[A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+',
            r'[A-Z][a-z]+(?:-[A-Z][a-z]+)+'
        ]
        
        for pattern in name_patterns:
            names = re.findall(pattern, query)
            if names:
                return max(names, key=len).strip()
        return None


class ChunkExtractor:
    """
    Responsible for extracting and grouping text content from HTML sources into manageable text chunks.
    
    - Cleans the HTML to extract readable text.
    - Splits the text into sentences using BlingFire.
    - Groups sentences into chunks of a specified size (e.g., 3 sentences per chunk).
    - Removes duplicates to avoid repetition.
    
    This helps reduce the complexity of feeding raw HTML directly into LLMs by structuring the data.
    """
    
    def __init__(self, sentence_group_size: int):
        self.sentence_group_size = sentence_group_size

    def _extract_chunks(self, interaction_id: str, html_source: str) -> Tuple[str, List[str]]:
        """
        Extract text chunks from a single HTML source.

        Steps:
        - Parse HTML with BeautifulSoup to remove tags and get raw text.
        - Split text into sentences and group them into chunks.
        
        Args:
            interaction_id: A unique identifier for the current interaction or query.
            html_source: HTML content of a webpage.
        
        Returns:
            A tuple (interaction_id, chunks) where chunks is a list of text chunks.
        """
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)

        if not text:
            return interaction_id, [""]

        _, offsets = text_to_sentences_and_offsets(text)
        chunks = []

        current_chunk = []
        for start, end in offsets:
            sentence = text[start:end][:MAX_CONTEXT_SENTENCE_LENGTH]
            current_chunk.append(sentence)
            
            # Once we reach the specified group size, form a chunk
            if len(current_chunk) >= self.sentence_group_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # If there's a remainder that doesn't form a full group, add it too
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return interaction_id, chunks

    def extract_chunks(self, 
                      queries: List[str], 
                      batch_interaction_ids: List[str], 
                      batch_search_results: List[List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of search results and convert them into text chunks.
        
        Args:
            queries: A list of query strings.
            batch_interaction_ids: A list of unique identifiers for each query.
            batch_search_results: For each query, a list of search result dictionaries,
                                  each containing 'page_result' with HTML content.
        
        Returns:
            Two arrays: chunks (array of all unique chunks from all queries) and 
            chunk_interaction_ids (array mapping each chunk back to its interaction_id).
        """
        chunk_dictionary = defaultdict(list)
        chunk_len_dist = {}

        for idx, search_results in enumerate(batch_search_results):
            # For simplicity, we currently only process the top 5 results
            # to avoid handling excessively large data.
            for html_text in search_results[:5]:
                interaction_id, _chunks = self._extract_chunks(
                    interaction_id=batch_interaction_ids[idx],
                    html_source=html_text["page_result"]
                )
                length = len(_chunks)
                chunk_len_dist[length] = chunk_len_dist.get(length, 0) + 1
                chunk_dictionary[interaction_id].extend(_chunks)

        print("Chunk length distribution:", chunk_len_dist)

        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)
        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten the dictionary of chunks into arrays, removing duplicates.
        
        Args:
            chunk_dictionary: Mapping from interaction_id to a list of text chunks.
            
        Returns:
            chunks: A numpy array of unique text chunks.
            chunk_interaction_ids: A numpy array mapping each chunk to the correct interaction_id.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            seen = set()
            unique_chunks = []
            for chunk in _chunks:
                # Deduplicate based on a hash of the normalized chunk text
                chunk_hash = hash(chunk.strip().lower())
                if chunk_hash not in seen:
                    seen.add(chunk_hash)
                    unique_chunks.append(chunk)
            
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        return np.array(chunks), np.array(chunk_interaction_ids)


class KGQuerier:
    """
    Handles queries to a Knowledge Graph (KG) via a Mock API.
    
    - Uses EntityExtractor to identify relevant entities in the query.
    - Queries multiple KG endpoints (movie, finance, music, sports, open) and aggregates results.
    - Implements caching and retry mechanisms for reliability.
    """
    
    def __init__(self, api_endpoint: str = "http://127.0.0.1:8000"):
        self.api_endpoint = api_endpoint
        self.entity_extractor = EntityExtractor()
        self.cache = {}  # Simple cache for API responses to avoid repeated calls
        self.max_retries = 3
        self.timeout = 5
        
    def _make_api_call(self, endpoint: str, data: Dict) -> Any:
        """Make an API call to a given endpoint with given payload.
        
        Implements caching and retries to handle transient errors.
        
        Args:
            endpoint: The KG API endpoint (e.g., '/movie/get_movie_info').
            data: Payload dictionary for the request.
            
        Returns:
            Parsed JSON result if available, otherwise None.
        """
        cache_key = f"{endpoint}:{json.dumps(data, sort_keys=True)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                # Some endpoints require specific payload formats rather than just "query"
                # We handle these cases explicitly.
                if endpoint in ['/sports/nba/get_games_on_date', '/sports/soccer/get_games_on_date']:
                    payload = {
                        "date": data.get("date", ""),
                        "team_name": data.get("team_name")
                    }
                elif endpoint == '/sports/nba/get_play_by_play_data_by_game_ids':
                    payload = {"game_ids": data.get("game_ids", [])}
                else:
                    payload = {"query": data.get("query")}

                response = requests.post(
                    f"{self.api_endpoint}{endpoint}",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json().get("result")
                if result:
                    self.cache[cache_key] = result
                return result
            except requests.exceptions.RequestException as e:
                # If an error occurs, print and retry unless it's the last attempt
                print(f"API call error ({endpoint}, attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e:
                print(f"Unexpected error ({endpoint}): {str(e)}")
                return None

    def query_kg(self, query: str, query_time: str) -> Dict:
        """Query all relevant KG domains (movie, finance, music, sports, open).
        
        Args:
            query: The user query string.
            query_time: The time context of the query (e.g., "2021-09-01 12:34:56").
        
        Returns:
            A dictionary containing combined results from all KG domains.
        """
        kg_data = {}
        
        movie_result = self._query_movie_kg(query)
        if movie_result:
            kg_data['movie'] = movie_result
            
        finance_result = self._query_finance_kg(query)
        if finance_result:
            kg_data['finance'] = finance_result
            
        music_result = self._query_music_kg(query)
        if music_result:
            kg_data['music'] = music_result
            
        sports_result = self._query_sports_kg(query, query_time)
        if sports_result:
            kg_data['sports'] = sports_result
            
        open_result = self._query_open_kg(query)
        if open_result:
            kg_data['open'] = open_result

        return kg_data

    def _query_movie_kg(self, query: str) -> Dict:
        """Query movie-related KG endpoints for info on movies and people related to the query."""
        movie_data = {}

        # Get movie information
        movies = self._make_api_call('/movie/get_movie_info', {'query': query})
        if movies:
            movie_data['movies'] = movies

        # Get person information
        person = self._make_api_call('/movie/get_person_info', {'query': query})
        if person:
            movie_data['person'] = person

        # Extract year from query and retrieve year-based info if valid
        try:
            year = self.entity_extractor.extract_time(query)
            if year and year.isdigit():
                year_int = int(year)
                if 1990 <= year_int <= 2021:
                    year_info = self._make_api_call('/movie/get_year_info', {'query': str(year_int)})
                    if year_info:
                        movie_data['year'] = year_info
        except (ValueError, TypeError) as e:
            print(f"Year handling error: {str(e)}")

        return movie_data

    def _query_finance_kg(self, query: str) -> Dict:
        """Query finance-related KG endpoints for company info, tickers, market data, etc."""
        finance_data = {}
        
        company_names = self._make_api_call('/finance/get_company_name', {'query': query})
        if not company_names:
            return {}

        # Use the first matching company name
        company_name = company_names[0] if isinstance(company_names, list) else company_names
        ticker = self._make_api_call('/finance/get_ticker_by_name', {'query': company_name})
        if not ticker:
            return {}
            
        finance_data['company'] = company_name
        finance_data['ticker'] = ticker

        # Retrieve additional financial metrics if available
        market_cap = self._make_api_call('/finance/get_market_capitalization', {'query': ticker})
        if market_cap:
            finance_data['market_cap'] = market_cap

        pe_ratio = self._make_api_call('/finance/get_pe_ratio', {'query': ticker})
        if pe_ratio:
            finance_data['pe_ratio'] = pe_ratio

        eps = self._make_api_call('/finance/get_eps', {'query': ticker})
        if eps:
            finance_data['eps'] = eps

        info = self._make_api_call('/finance/get_info', {'query': ticker})
        if info:
            finance_data['company_info'] = info

        return finance_data

    def _query_music_kg(self, query: str) -> Dict:
        """Query music-related KG endpoints for artist and song data."""
        music_data = {}
        
        # Try artist search
        artists = self._make_api_call('/music/search_artist_entity_by_name', {'query': query})
        if artists:
            artist = artists[0] if isinstance(artists, list) else artists
            music_data['artist'] = artist
            
            # Retrieve additional artist details
            birth_place = self._make_api_call('/music/get_artist_birth_place', {'query': artist})
            if birth_place:
                music_data['birth_place'] = birth_place
                
            birth_date = self._make_api_call('/music/get_artist_birth_date', {'query': artist})
            if birth_date:
                music_data['birth_date'] = birth_date
                
            awards = self._make_api_call('/music/grammy_get_award_count_by_artist', {'query': artist})
            if awards:
                music_data['grammy_awards'] = awards
                
            works = self._make_api_call('/music/get_artist_all_works', {'query': artist})
            if works:
                music_data['works'] = works

        # Try song search
        songs = self._make_api_call('/music/search_song_entity_by_name', {'query': query})
        if songs:
            song = songs[0] if isinstance(songs, list) else songs
            if 'songs' not in music_data:
                music_data['songs'] = []
            
            song_data = {'name': song}
            
            author = self._make_api_call('/music/get_song_author', {'query': song})
            if author:
                song_data['author'] = author
                
            release_date = self._make_api_call('/music/get_song_release_date', {'query': song})
            if release_date:
                song_data['release_date'] = release_date
                
            music_data['songs'].append(song_data)

        return music_data

    def _query_sports_kg(self, query: str, query_time: str) -> Dict:
        """Query sports-related KG endpoints, handling date formatting and potential team extraction."""
        sports_data = {}

        # Parse query_time to a date if possible
        try:
            query_date = datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S")
            formatted_date = query_date.strftime("%Y-%m-%d")
        except ValueError:
            try:
                query_date = datetime.strptime(query_time, "%Y-%m-%d")
                formatted_date = query_time
            except ValueError:
                formatted_date = query_time

        # Attempt to extract a team name from the query
        team_name = self.entity_extractor.extract_company_name(query)
        
        # Query NBA games on a given date
        nba_request = {
            "date": formatted_date,
            "team_name": team_name
        }
        nba_games = self._make_api_call('/sports/nba/get_games_on_date', nba_request)
        if nba_games:
            sports_data['nba'] = nba_games
            
            # If we have game IDs, try to get play-by-play data
            if isinstance(nba_games, list):
                game_ids = [game.get('game_id') for game in nba_games if game.get('game_id')]
                if game_ids:
                    play_data = {"game_ids": game_ids}
                    play_by_play = self._make_api_call('/sports/nba/get_play_by_play_data_by_game_ids', play_data)
                    if play_by_play:
                        sports_data['nba_play_by_play'] = play_by_play

        # Query Soccer data
        soccer_request = {
            "date": formatted_date,
            "team_name": team_name
        }
        soccer_games = self._make_api_call('/sports/soccer/get_games_on_date', soccer_request)
        if soccer_games:
            sports_data['soccer'] = soccer_games

        return sports_data

    def _query_open_kg(self, query: str) -> Dict:
        """Query open-domain KG endpoints, typically returning general information."""
        entities = self._make_api_call('/open/search_entity_by_name', {'query': query})
        if not entities:
            return {}

        # Use the first returned entity to get details
        entity_details = self._make_api_call('/open/get_entity', {'query': entities[0]})
        return entity_details if entity_details else {}


class Task2RAGModel:
    """
    Implements a Retrieval-Augmented Generation (RAG) model for Task 2.
    
    Features:
    - First attempts to answer using KG data alone.
    - If KG data is insufficient or uncertain, fallback to web (search results) data.
    - If both are insufficient, answer "I don't know".
    - Uses a sentence transformer to compute embeddings for ranking relevance of web content.
    - Uses vLLM or a server-based model to generate final answers.
    """
    
    def __init__(self, llm_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 is_server: bool = False,
                 vllm_server: Optional[str] = None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor(sentence_group_size=NUM_CHUNK_PER_SENTENCE)
        self.chunk_len_limit_count = 0
        self.kg_querier = KGQuerier()

    def initialize_models(self, llm_name: str, is_server: bool, vllm_server: Optional[str]):
        """Initialize the language model (LLM) and embedding model.
        
        If is_server=True, use an external OpenAI-style API for inference.
        Otherwise, load the model locally with vLLM.
        Also initializes a SentenceTransformer for embedding calculation.
        """
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # Use a local server that mimics the OpenAI Chat API
            self.llm_client = OpenAI(
                api_key="EMPTY",
                base_url=self.vllm_server,
            )
        else:
            print('Loading LLM...')
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=False,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()
            print("LLM loaded!")

        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def calculate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute normalized embeddings for a list of sentences."""
        return self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
        )

    def get_batch_size(self) -> int:
        """Return batch size used for evaluation."""
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """Generate answers for a batch of queries using a two-stage strategy:
        
        Stage 1:
        - Query the KG for an answer.
        - If a confident KG-only answer is found, return it.
        
        Stage 2:
        - If KG is insufficient or uncertain, extract web chunks, rank them by similarity,
          and try to form an answer using both KG and web context.
        
        If neither approach yields a confident answer, return 'I don't know'.
        
        Args:
            batch: Dictionary containing:
                   "interaction_id": List of unique IDs
                   "query": List of queries
                   "search_results": List of web search results
                   "query_time": List of timestamps for each query

        Returns:
            A list of answers corresponding to each query.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
    
        answers = []
        for idx, (interaction_id, query, search_results, query_time) in enumerate(
            zip(batch_interaction_ids, queries, batch_search_results, query_times)):
            
            # Stage 1: Attempt a KG-only response
            kg_data = self.kg_querier.query_kg(query, query_time)
            if kg_data:
                kg_response = self._generate_kg_response(query, kg_data)
                if kg_response and self._is_response_confident(kg_response):
                    answers.append(kg_response)
                    continue
    
            # Stage 2: Use web data since KG response was uncertain
            chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                [query], [interaction_id], [search_results]
            )
            
            if len(chunks) > 0:
                chunk_embeddings = self.calculate_embeddings(chunks)
                query_embedding = self.calculate_embeddings([query])[0]
                
                # Rank chunks by cosine similarity and select top N
                cosine_scores = np.dot(chunk_embeddings, query_embedding)
                top_indices = cosine_scores.argsort()[::-1][:NUM_CONTEXT_SENTENCES]
                web_results = chunks[top_indices].tolist()
                
                # Integrate KG data as well for a more comprehensive context
                context = {
                    'web_results': web_results,
                    'kg_data': kg_data
                }
                
                prompt = self._format_prompts([query], [query_time], [context])[0]
                web_response = self._generate_llm_response(prompt)
                
                if web_response and self._is_response_confident(web_response):
                    answers.append(web_response)
                    continue
            
            # If no confident answer found
            answers.append("I don't know")
    
        return answers
    
    def _is_response_confident(self, response: str) -> bool:
        """Check if a generated response appears confident and informative.

        We consider a response confident if it:
        - Does not contain uncertainty indicators like "not sure", "maybe", etc.
        - Is at least a few words long.
        - Potentially contains specific details (numbers, capitalized words).
        
        Args:
            response: The LLM-generated answer.

        Returns:
            True if confident, False otherwise.
        """
        uncertainty_indicators = [
            "i'm not sure", "might be", "could be", "possibly", "maybe", "i think",
            "probably", "appears to be", "seems like", "unclear", "don't know",
            "can't determine", "cannot determine", "uncertain", "insufficient information",
            "not enough information", "ambiguous"
        ]
        
        response_lower = response.lower()
        for indicator in uncertainty_indicators:
            if indicator in response_lower:
                return False
        
        # Check minimal length
        if len(response.split()) < 3:
            return False
        
        # Check for specificity (digits or capitalized words)
        has_specific_details = any(char.isdigit() for char in response) or \
                               any(word[0].isupper() for word in response.split())
        
        return has_specific_details

    def _generate_kg_response(self, query: str, kg_data: Dict) -> Optional[str]:
        """Generate a response using only KG data if available."""
        if not kg_data:
            return None

        kg_context = self._format_kg_data(kg_data)
        if not kg_context.strip():
            return None

        prompt = [
            {"role": "system", "content": "You are a knowledgeable assistant. Use the provided knowledge graph data to answer the question accurately and concisely."},
            {"role": "user", "content": f"Knowledge Graph Data:\n{kg_context}\n\nQuestion: {query}"}
        ]

        return self._generate_llm_response(prompt)

    def _generate_llm_response(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Generate the final LLM response, adding instructions for uncertainty handling.
        
        If using a server API (OpenAI style), send the messages and get a response.
        If using local LLM via vLLM, generate directly.
        
        Args:
            prompt: Either a string or a list of role-content messages (for Chat API).
            
        Returns:
            The generated answer text. If an error occurs, returns "I don't know".
        """
        try:
            if self.is_server:
                # Add a system message encouraging explicit uncertainty if not confident
                response = self.llm_client.chat.completions.create(
                    model=self.llm_name,
                    messages=prompt + [{
                        "role": "system",
                        "content": "If you are not completely confident in your answer, "
                                   "indicate your uncertainty clearly. Only provide definitive "
                                   "answers when you have strong supporting evidence."
                    }],
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    max_tokens=75,
                )
                answer = response.choices[0].message.content.strip()
            else:
                # If using local vLLM model
                if isinstance(prompt, str):
                    uncertainty_prompt = prompt + "\nNote: If you are not completely confident, " \
                                                   "indicate your uncertainty clearly. Only provide definitive " \
                                                   "answers when you have strong supporting evidence."
                else:
                    # Insert a system message with instructions if the prompt is a conversation
                    prompt.append({
                        "role": "system",
                        "content": "If you are not completely confident, indicate your "
                                   "uncertainty clearly. Only provide definitive answers when "
                                   "you have strong supporting evidence."
                    })
                    uncertainty_prompt = self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                
                response = self.llm.generate(
                    [uncertainty_prompt],
                    vllm.SamplingParams(
                        n=1,
                        top_p=0.9,
                        temperature=0.1,
                        skip_special_tokens=True,
                        max_tokens=75,
                    ),
                )[0]
                answer = response.outputs[0].text.strip()
    
            # Ensure answer does not exceed 75 tokens
            answer_tokens = answer.split()
            if len(answer_tokens) > 75:
                answer = ' '.join(answer_tokens[:75])
    
            return answer
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I don't know"

    def _format_kg_data(self, kg_data: Dict) -> str:
        """Convert KG data from various domains into a formatted text string.
        
        This helps the LLM understand the structured data by presenting it in a readable manner.
        
        Args:
            kg_data: Dictionary containing results from various KG endpoints.
        
        Returns:
            A formatted, human-readable string summarizing KG data.
        """
        formatted_text = []
        
        for domain, data in kg_data.items():
            if not data:
                continue
                
            formatted_text.append(f"\n## {domain.title()} Information:")
            
            # Format data domain-by-domain
            if domain == 'movie':
                if 'movies' in data:
                    formatted_text.append("Movies:")
                    for movie in data['movies']:
                        formatted_text.extend([
                            f"Title: {movie.get('title', '')}",
                            f"Release Date: {movie.get('release_date', '')}",
                            f"Rating: {movie.get('rating', '')}",
                            f"Original Language: {movie.get('original_language', '')}",
                            f"Budget: ${movie.get('budget', 0):,}",
                            f"Revenue: ${movie.get('revenue', 0):,}"
                        ])
                        if movie.get('oscar_awards'):
                            formatted_text.append("Oscar Awards:")
                            for award in movie['oscar_awards']:
                                formatted_text.append(
                                    f"- {award.get('year_ceremony', '')}: {award.get('category', '')}"
                                    f" ({'Won' if award.get('winner') else 'Nominated'})"
                                )
                        formatted_text.append("")
                
                if 'person' in data:
                    formatted_text.append("People:")
                    for person in data['person']:
                        formatted_text.extend([
                            f"Name: {person.get('name', '')}",
                            f"Birthday: {person.get('birthday', '')}"
                        ])
                        if person.get('oscar_awards'):
                            formatted_text.append("Oscar Awards:")
                            for award in person['oscar_awards']:
                                formatted_text.append(
                                    f"- {award.get('year_ceremony', '')}: {award.get('category', '')}"
                                    f" ({'Won' if award.get('winner') else 'Nominated'})"
                                )
                        if person.get('acted_movies'):
                            formatted_text.append("Acted in:")
                            formatted_text.append(f"- {len(person['acted_movies'])} movies")
                        if person.get('directed_movies'):
                            formatted_text.append("Directed:")
                            formatted_text.append(f"- {len(person['directed_movies'])} movies")
                        formatted_text.append("")
                
                if 'year' in data:
                    year_info = data['year']
                    if year_info.get('oscar_awards'):
                        formatted_text.append("Oscar Awards this year:")
                        for award in year_info['oscar_awards']:
                            formatted_text.append(
                                f"- {award.get('category', '')}: {award.get('name', '')}"
                                f" for '{award.get('film', '')}'"
                                f" ({'Won' if award.get('winner') else 'Nominated'})"
                            )
                    if year_info.get('movie_list'):
                        formatted_text.append(f"Total movies this year: {len(year_info['movie_list'])}")
            
            elif domain == 'finance':
                formatted_text.extend([
                    f"Company: {data.get('company', '')}",
                    f"Ticker: {data.get('ticker', '')}",
                    f"Market Cap: ${data.get('market_cap', 0):,}",
                    f"P/E Ratio: {data.get('pe_ratio', '')}",
                    f"EPS: ${data.get('eps', '')}"
                ])
                if data.get('company_info'):
                    info = data['company_info']
                    formatted_text.append("Company Information:")
                    for key, value in info.items():
                        formatted_text.append(f"- {key}: {value}")
            
            elif domain == 'music':
                if 'artist' in data:
                    formatted_text.extend([
                        f"Artist: {data['artist']}",
                        f"Birth Place: {data.get('birth_place', '')}",
                        f"Birth Date: {data.get('birth_date', '')}",
                        f"Grammy Awards: {data.get('grammy_awards', '')}"
                    ])
                    if data.get('works'):
                        formatted_text.append(f"Total Works: {len(data['works'])}")
                
                if 'songs' in data:
                    formatted_text.append("Songs:")
                    for song in data['songs']:
                        formatted_text.extend([
                            f"Title: {song.get('name', '')}",
                            f"Author: {song.get('author', '')}",
                            f"Release Date: {song.get('release_date', '')}"
                        ])
            
            elif domain == 'sports':
                if 'nba' in data:
                    formatted_text.append("NBA Games:")
                    for game in data['nba']:
                        formatted_text.extend([
                            f"Home: {game.get('team_name_home', '')}, Score: {game.get('pts_home', '')}",
                            f"Away: {game.get('team_name_away', '')}, Score: {game.get('pts_away', '')}",
                            f"Result: {game.get('wl_home', '')} (Home), {game.get('wl_away', '')} (Away)",
                            ""
                        ])
                
                if 'soccer' in data:
                    formatted_text.append("Soccer Games:")
                    for game in data['soccer']:
                        formatted_text.extend([
                            f"Teams: {game.get('team', '')} vs {game.get('opponent', '')}",
                            f"Result: {game.get('result', '')}",
                            f"Goals: {game.get('GF', '')}",
                            f"Venue: {game.get('venue', '')}",
                            f"Captain: {game.get('Captain', '')}",
                            ""
                        ])
            
            elif domain == 'open':
                if 'summary_text' in data:
                    formatted_text.append(f"Summary: {data['summary_text']}")
                if 'summary_structured' in data:
                    formatted_text.append("Structured Information:")
                    for key, value in data['summary_structured'].items():
                        formatted_text.append(f"- {key}: {value}")
                
        return '\n'.join(formatted_text)

    def _format_prompts(self, queries: List[str], query_times: List[str], 
                        batch_contexts: List[Dict[str, Any]]) -> List[Union[str, List[Dict[str, str]]]]:
        """Format prompts for the LLM using both web results and KG data.
        
        Combines:
        - A system message describing the instructions.
        - A user message containing the query, current time, and references.
        
        Args:
            queries: List of user queries.
            query_times: List of query times for each query.
            batch_contexts: List of context dictionaries with 'web_results' and 'kg_data'.

        Returns:
            A list of formatted prompts. If using server mode, returns role-content messages.
            If using local vLLM, returns a template-processed prompt string.
        """
        system_prompt = """You are provided with web search results and structured knowledge graph data from multiple domains.
    Your task is to provide accurate and concise answers by synthesizing information from both sources.
    Only provide an answer if you can find clear supporting evidence from either the web search results or knowledge graph data.
    If neither the web results nor the knowledge graph data contain relevant information to answer the question, you must respond with 'I don't know'.
    Do not make assumptions or provide speculative answers when the data is insufficient.
    Focus on the most relevant and up-to-date information available in the provided context."""
    
        formatted_prompts = []
        for idx, query in enumerate(queries):
            query_time = query_times[idx]
            context = batch_contexts[idx]
            
            references = self._format_context_references(context)
            
            user_message = (
                f"{references}\n------\n\n"
                f"Using the above information, answer this question:\n"
                f"Current Time: {query_time}\n"
                f"Question: {query}\n"
            )
    
            if self.is_server:
                # For server-based model, return list of message dictionaries
                formatted_prompts.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ])
            else:
                # For local model, use tokenizer's chat template if supported
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
    
        return formatted_prompts
    
    def _format_context_references(self, context: Dict[str, Any]) -> str:
        """Format the web and KG references into a readable text block.
        
        Args:
            context: Dictionary with 'web_results' and 'kg_data'.
            
        Returns:
            A string containing references from both web data and KG.
            Truncated if it exceeds MAX_CONTEXT_REFERENCES_LENGTH.
        """
        references = ""
        
        # Format web content
        if context['web_results']:
            references += "# Web References\n"
            for idx, snippet in enumerate(context['web_results'], 1):
                references += f"{idx}. {snippet.strip()}\n"
    
        # Format knowledge graph data
        if context['kg_data']:
            references += "\n# Knowledge Graph Data\n"
            references += self._format_kg_data(context['kg_data'])
    
        # Ensure the references string does not exceed the defined max length
        return references[:MAX_CONTEXT_REFERENCES_LENGTH]
