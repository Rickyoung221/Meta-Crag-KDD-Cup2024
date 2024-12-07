import json
import re
import gc
import os
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

# the code handle 50 pages
#### CONFIG PARAMETERS ---
NUM_CHUNK_PER_SENTENCE = 3            # Number of sentences grouped into one chunk
NUM_CONTEXT_SENTENCES = 20            # Number of top relevant chunks to include as context
MAX_CONTEXT_SENTENCE_LENGTH = 1000    # Maximum length of each sentence in characters
MAX_CONTEXT_REFERENCES_LENGTH = 4000  # Maximum total length of references provided to the LLM
AICROWD_SUBMISSION_BATCH_SIZE = 1     # Batch size for inference submission
VLLM_TENSOR_PARALLEL_SIZE = 1         # Tensor parallelism for vLLM
VLLM_GPU_MEMORY_UTILIZATION = 0.85    # GPU memory utilization ratio for vLLM inference
SENTENCE_TRANSFORMER_BATCH_SIZE = 32  # Batch size for sentence embedding calculation

MAX_PAGES_PER_QUERY = 50     # Maximum number of web pages to process per query
RELEVANCE_THRESHOLD = 0.2    # Minimum relevance score threshold for a page to be considered
BATCH_SIZE = 30              # Batch processing size for embedding calculations
MAX_CHUNKS_PER_PAGE = 50     # Maximum number of chunks extracted per page
MAX_TOTAL_CHUNKS = 200       # Maximum total number of chunks per query after filtering
#### CONFIG PARAMETERS END---

class EntityExtractor:
    """
    Extracts specific entities (time, company name, person name) from the query string.
    Useful for providing additional context or directing certain KG queries.
    """

    @staticmethod
    def extract_time(query: str) -> Optional[str]:
        """
        Extract a time reference from the query, such as a year or a date.
        
        Tries multiple patterns:
        - YYYY format (e.g., 1999, 2020)
        - Abbreviated years like '23
        - Full date formats: YYYY-MM-DD or "Month Day, Year"
        
        Returns:
            A string representing the first detected time, or None if not found.
        """
        year_patterns = [
            r'\b(?:19|20)\d{2}\b',  # Matches years 1900-2099
            r'\'\d{2}\b',           # Matches abbreviated year like '23
            r'\b\d{4}\b'            # Matches any four-digit number
        ]
        
        for pattern in year_patterns:
            years = re.findall(pattern, query)
            if years:
                year = years[0]
                # Handle abbreviated years like '23
                if '\'' in year:
                    year = '20' + year[1:] if int(year[1:]) < 50 else '19' + year[1:]
                # If we got a two-digit year from above logic, convert it properly
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                return year

        # Look for full date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            if dates:
                return dates[0]
        
        return None

    @staticmethod
    def extract_company_name(query: str) -> Optional[str]:
        """
        Extract a potential company name from the query.
        
        Uses patterns for known company suffixes and capitalized words to identify a candidate.
        
        Returns:
            The longest matched company name or None if no match.
        """
        suffixes = r'\b(?:Inc|Corp|Ltd|LLC|Company|Co|Group|Holdings|Technology|Tech|Industries|International)\b'
        prefixes = r'\b(?:The)\b'
        
        company_patterns = [
            rf'(?:{prefixes}\s+)?[A-Z][A-Za-z0-9\s&\'-]+(?:\s+{suffixes})?',
            rf'[A-Z][A-Za-z0-9\s&\'-]+(?:\s+{suffixes})',
            r'[A-Z]{2,}(?:\s+[A-Z]{2,})*'
        ]
        
        for pattern in company_patterns:
            companies = re.findall(pattern, query)
            if companies:
                return max(companies, key=len).strip()
        return None

    @staticmethod
    def extract_person_name(query: str) -> Optional[str]:
        """
        Extract a potential person's name from the query.
        
        Tries patterns that match first and last names, possibly with middle initials or particles.
        
        Returns:
            The longest matched name if found, otherwise None.
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


class RelevanceScorer:
    """
    Scores a webpage's relevance to a given query by using semantic similarity (embeddings),
    keyword frequency, and information density measures.
    Only pages exceeding a certain threshold will be considered for chunk extraction.
    """

    def __init__(self, sentence_transformer: SentenceTransformer):
        self.sentence_model = sentence_transformer
        self.cache = {}

    def calculate_page_relevance(self, html_content: str, query: str) -> float:
        """
        Calculate a relevance score for a given HTML page and query.

        Steps:
            - Extract text from HTML
            - Compute semantic similarity to the query
            - Compute keyword frequency score
            - Compute an information density score
            - Combine scores into a weighted final score

        Returns:
            A float representing the relevance score.
        """
        try:
            cache_key = f"{hash(html_content)}_{hash(query)}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            soup = BeautifulSoup(html_content, "lxml")
            text = soup.get_text(" ", strip=True)
            
            if not text:
                return 0.0

            # Compute embeddings and semantic similarity
            text_embedding = self.sentence_model.encode(text[:512], normalize_embeddings=True)
            query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)
            semantic_score = float(np.dot(text_embedding, query_embedding))

            # Keyword score
            keyword_score = self._calculate_keyword_score(text, query)
            
            # Density score
            density_score = self._calculate_density_score(text)
            
            # Weighted combination
            final_score = (semantic_score * 0.5 + keyword_score * 0.3 + density_score * 0.2)
            
            self.cache[cache_key] = final_score
            return final_score

        except Exception as e:
            print(f"Error calculating page relevance: {str(e)}")
            return 0.0

    def _calculate_keyword_score(self, text: str, query: str) -> float:
        """
        Calculate how frequently query terms appear in the text.
        
        Returns:
            A frequency-based score.
        """
        try:
            text_lower = text.lower()
            query_terms = query.lower().split()
            
            term_frequencies = []
            for term in query_terms:
                frequency = text_lower.count(term) / max(1, len(text_lower.split()))
                term_frequencies.append(frequency)
            
            return sum(term_frequencies) / len(query_terms) if query_terms else 0.0
        except Exception as e:
            print(f"Error calculating keyword score: {str(e)}")
            return 0.0

    def _calculate_density_score(self, text: str) -> float:
        """
        Calculate information density by counting occurrences of:
            - numbers
            - dates
            - names
            - URLs
            - Emails

        Returns:
            A density score as a float.
        """
        try:
            patterns = {
                'numbers': r'\b\d+(?:\.\d+)?\b',
                'dates': r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',
                'names': r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
                'urls': r'http[s]?://(?:[^\s]+)',
                'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            }
            
            words = text.split()
            total_words = len(words)
            if total_words == 0:
                return 0.0
                
            densities = []
            for pattern in patterns.values():
                matches = len(re.findall(pattern, text))
                density = matches / total_words
                densities.append(density)
            
            return sum(densities) / len(densities)
        except Exception as e:
            print(f"Error calculating density score: {str(e)}")
            return 0.0


class ChunkExtractor:
    """
    Extracts text chunks from filtered web pages:
    - Convert HTML to text
    - Split into sentences
    - Group sentences into chunks
    - Limit total chunks
    - De-duplicate and score chunks to select top ones
    """

    def __init__(self, sentence_group_size: int):
        self.sentence_group_size = sentence_group_size
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.relevance_scorer = RelevanceScorer(self.sentence_model)
        self.processed_pages = 0

    def _extract_chunks(self, interaction_id: str, html_source: str) -> Tuple[str, List[str]]:
        """
        Extract chunks from a single HTML source:
            - Get text
            - Split into sentences
            - Group sentences into chunks of size self.sentence_group_size
            - Limit to MAX_CHUNKS_PER_PAGE

        Returns:
            A tuple of (interaction_id, list_of_chunks)
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
            
            if len(current_chunk) >= self.sentence_group_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return interaction_id, chunks[:MAX_CHUNKS_PER_PAGE]

    def extract_chunks(self, 
                      queries: List[str], 
                      batch_interaction_ids: List[str], 
                      batch_search_results: List[List[Dict]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query:
            - Write out the first 50 pages before filtering for inspection
            - Compute relevance scores for pages
            - Filter pages below threshold
            - Extract chunks from top pages until limits are reached
            - Return all chosen chunks and their interaction_ids as arrays
        """
        chunk_dictionary = defaultdict(list)
        chunk_len_dist = {}
        page_scores = defaultdict(list)

        output_dir = "web_pages_before_filtering"
        os.makedirs(output_dir, exist_ok=True)

        for idx, search_results in enumerate(batch_search_results):
            query = queries[idx]
            self.processed_pages = 0
            
            # Create a separate directory for each query
            query_output_dir = os.path.join(output_dir, f"query_{batch_interaction_ids[idx]}")
            os.makedirs(query_output_dir, exist_ok=True)

            # Write the first 50 pages to files
            for i, html_text in enumerate(search_results[:50]):
                with open(os.path.join(query_output_dir, f"page_{i+1}.html"), "w", encoding="utf-8") as f:
                    f.write(html_text["page_result"])

            # Score pages and filter
            scored_results = []
            for html_text in search_results:
                if self.processed_pages >= MAX_PAGES_PER_QUERY:
                    break
                    
                score = self.relevance_scorer.calculate_page_relevance(
                    html_text["page_result"], 
                    query
                )
                if score >= RELEVANCE_THRESHOLD:
                    scored_results.append((html_text, score))
                self.processed_pages += 1

            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            total_chunks = 0
            for html_text, score in scored_results:
                if total_chunks >= MAX_TOTAL_CHUNKS:
                    break
                    
                interaction_id, _chunks = self._extract_chunks(
                    interaction_id=batch_interaction_ids[idx],
                    html_source=html_text["page_result"]
                )
                
                length = len(_chunks)
                chunk_len_dist[length] = chunk_len_dist.get(length, 0) + 1
                
                chunk_dictionary[interaction_id].extend(_chunks)
                page_scores[interaction_id].extend([score] * len(_chunks))
                
                total_chunks += len(_chunks)

        print(f"Processed {self.processed_pages} pages")
        print("Chunk length distribution:", chunk_len_dist)

        # Flatten chunks, deduplicate, and choose top scoring ones
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary, page_scores)
        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, 
                       chunk_dictionary: Dict[str, List[str]],
                       page_scores: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten and deduplicate chunks across pages:
            - Sort chunks by score
            - Remove duplicates keeping highest score
            - Limit total number of chunks

        Returns:
            arrays of chunks and corresponding interaction_ids
        """
        chunks = []
        chunk_interaction_ids = []
        seen_content = {}

        for interaction_id, _chunks in chunk_dictionary.items():
            scores = page_scores[interaction_id]
            chunk_score_pairs = list(zip(_chunks, scores))
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for chunk, score in chunk_score_pairs:
                chunk_hash = hash(chunk.strip().lower())
                if chunk_hash not in seen_content or seen_content[chunk_hash][1] < score:
                    seen_content[chunk_hash] = (chunk, score)

        sorted_chunks = sorted(seen_content.values(), key=lambda x: x[1], reverse=True)
        
        for chunk, _ in sorted_chunks[:MAX_TOTAL_CHUNKS]:
            chunks.append(chunk)
            # Arbitrarily assign the first key from the dictionary as interaction_id
            # In practice, this could be improved if multiple queries are processed at once.
            chunk_interaction_ids.append(next(iter(chunk_dictionary.keys())))

        return np.array(chunks), np.array(chunk_interaction_ids)


class KGQuerier:
    """
    Queries a mock Knowledge Graph API for structured data.
    Tries multiple domains: movie, finance, music, sports, open.
    Uses caching and retry logic.
    """

    def __init__(self, api_endpoint: str = "http://127.0.0.1:8000"):
        self.api_endpoint = api_endpoint
        self.entity_extractor = EntityExtractor()
        self.cache = {}
        self.max_retries = 3
        self.timeout = 5
        
    def _make_api_call(self, endpoint: str, data: Dict) -> Any:
        """
        Make POST requests to the KG API and handle caching and retries.
        
        Returns:
            The 'result' field from the JSON response or None if errors occur.
        """
        cache_key = f"{endpoint}:{json.dumps(data, sort_keys=True)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                # Some endpoints require special payload structure
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
                print(f"API call error ({endpoint}, attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e:
                print(f"Unexpected error ({endpoint}): {str(e)}")
                return None

    def query_kg(self, query: str, query_time: str) -> Dict:
        """
        Query all KG endpoints and aggregate results.
        
        Returns a dictionary of domain-specific data.
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
        """
        Query movie-related KG:
            - movie info
            - person info
            - year info if applicable
        """
        movie_data = {}

        movies = self._make_api_call('/movie/get_movie_info', {'query': query})
        if movies:
            movie_data['movies'] = movies

        person = self._make_api_call('/movie/get_person_info', {'query': query})
        if person:
            movie_data['person'] = person

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
        """
        Query finance-related KG endpoints:
            - Get company name
            - Get ticker
            - Market cap, P/E ratio, EPS, company info if available
        """
        finance_data = {}
        
        company_names = self._make_api_call('/finance/get_company_name', {'query': query})
        if not company_names:
            return {}

        company_name = company_names[0] if isinstance(company_names, list) else company_names
        ticker = self._make_api_call('/finance/get_ticker_by_name', {'query': company_name})
        if not ticker:
            return {}
            
        finance_data['company'] = company_name
        finance_data['ticker'] = ticker

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
        """
        Query music-related KG:
            - Artist info, birth place/date, awards, works
            - Song info, author, release date
        """
        music_data = {}
        
        artists = self._make_api_call('/music/search_artist_entity_by_name', {'query': query})
        if artists:
            artist = artists[0] if isinstance(artists, list) else artists
            music_data['artist'] = artist
            
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
        """
        Query sports-related KG:
            - NBA or soccer games on a given date
            - Play-by-play data if available
        """
        sports_data = {}

        try:
            query_date = datetime.strptime(query_time, "%Y-%m-%d %H:%M:%S")
            formatted_date = query_date.strftime("%Y-%m-%d")
        except ValueError:
            try:
                query_date = datetime.strptime(query_time, "%Y-%m-%d")
                formatted_date = query_time
            except ValueError:
                formatted_date = query_time

        team_name = self.entity_extractor.extract_company_name(query)
        
        nba_request = {
            "date": formatted_date,
            "team_name": team_name
        }
        nba_games = self._make_api_call('/sports/nba/get_games_on_date', nba_request)
        if nba_games:
            sports_data['nba'] = nba_games
            if isinstance(nba_games, list):
                game_ids = [game.get('game_id') for game in nba_games if game.get('game_id')]
                if game_ids:
                    play_data = {"game_ids": game_ids}
                    play_by_play = self._make_api_call('/sports/nba/get_play_by_play_data_by_game_ids', play_data)
                    if play_by_play:
                        sports_data['nba_play_by_play'] = play_by_play

        soccer_request = {
            "date": formatted_date,
            "team_name": team_name
        }
        soccer_games = self._make_api_call('/sports/soccer/get_games_on_date', soccer_request)
        if soccer_games:
            sports_data['soccer'] = soccer_games

        return sports_data

    def _query_open_kg(self, query: str) -> Dict:
        """
        Query an open-domain KG endpoint.
        """
        entities = self._make_api_call('/open/search_entity_by_name', {'query': query})
        if not entities:
            return {}

        entity_details = self._make_api_call('/open/get_entity', {'query': entities[0]})
        return entity_details if entity_details else {}


class Task3RAGModel:
    """
    A Retrieval-Augmented Generation (RAG) model pipeline:
    - Handles up to 50 pages per query
    - Filters and extracts chunks
    - Integrates KG data
    - Prompts an LLM for an answer
    - If insufficient info, returns "I don't know"
    """

    def __init__(self, llm_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 is_server: bool = False,
                 vllm_server: Optional[str] = None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor(sentence_group_size=NUM_CHUNK_PER_SENTENCE)
        self.kg_querier = KGQuerier()
        self.page_count = 0
        self.processed_chunks = 0

    def initialize_models(self, llm_name: str, is_server: bool, vllm_server: Optional[str]):
        """
        Initialize LLM and embedding model.
        If is_server is True, uses an OpenAI-compatible server.
        Otherwise, loads LLM locally using vLLM.
        """
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
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

        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generate answers for a batch of queries.

        Steps:
            - Extract and filter web chunks
            - Calculate embeddings and select top chunks
            - Query KG data
            - Combine into a single prompt
            - Generate LLM response
            - Validate and clean the final answer

        Returns:
            A list of answers, one per query.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        answers = []
        for idx, (interaction_id, query, search_results, query_time) in enumerate(
            zip(batch_interaction_ids, queries, batch_search_results, query_times)):

            self.page_count = 0
            self.processed_chunks = 0
            
            try:
                # Extract top chunks from pages
                chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
                    [query], [interaction_id], [search_results[:MAX_PAGES_PER_QUERY]]
                )
                
                if len(chunks) > 0:
                    # Compute similarity and select top relevant chunks
                    chunk_embeddings = self.calculate_embeddings(chunks)
                    query_embedding = self.calculate_embeddings([query])[0]
                    cosine_scores = np.dot(chunk_embeddings, query_embedding)
                    top_indices = cosine_scores.argsort()[::-1][:NUM_CONTEXT_SENTENCES]
                    web_results = chunks[top_indices].tolist()
                    self.processed_chunks = len(web_results)
                else:
                    web_results = []

                # Query KG
                kg_data = self.kg_querier.query_kg(query, query_time)

                # Format context and prompt
                context = self._format_context(web_results, kg_data)
                prompt = self._create_prompt(query, query_time, context)

                # Generate LLM response
                answer = self._generate_llm_response(prompt)

                # Validate and clean answer
                final_answer = self._validate_and_clean_answer(answer, context)
                answers.append(final_answer)
                
                print(f"Processed {self.page_count} pages, {self.processed_chunks} chunks for query: {query[:50]}...")
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                answers.append("I don't know")

        return answers

    def calculate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate normalized sentence embeddings in batches.
        """
        return self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
        )

    def _format_context(self, web_results: List[str], kg_data: Dict) -> str:
        """
        Combine selected web results and KG data into a context string.
        Truncate if it exceeds MAX_CONTEXT_REFERENCES_LENGTH.
        """
        context_parts = []
        
        if web_results:
            context_parts.append("\n# Web Content:")
            for idx, result in enumerate(web_results, 1):
                context_parts.append(f"{idx}. {result.strip()}")

        if kg_data:
            context_parts.append("\n# Knowledge Graph Information:")
            for domain, data in kg_data.items():
                if data:
                    context_parts.append(f"\n## {domain.title()}:")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            context_parts.append(f"{key}: {value}")
                    elif isinstance(data, list):
                        for item in data:
                            context_parts.append(str(item))
                    else:
                        context_parts.append(str(data))

        return "\n".join(context_parts)[:MAX_CONTEXT_REFERENCES_LENGTH]

    def _create_prompt(self, query: str, query_time: str, context: str) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM with system and user messages.
        System message provides instructions, user message includes context and question.
        """
        system_prompt = """You are a highly knowledgeable assistant tasked with answering questions using provided context 
from both web sources and knowledge graphs. Follow these guidelines:
1. Always verify information across multiple sources when available
2. Prioritize recent and relevant information
3. If sources conflict, mention the discrepancy and use the most reliable source
4. If insufficient information is available, respond with 'I don't know'
5. Keep answers concise but comprehensive (maximum 75 words)
6. Do not speculate or add information beyond what's provided in the context"""

        user_message = (
            f"Context Information:\n{context}\n\n"
            f"Current Time: {query_time}\n"
            f"Question: {query}\n\n"
            "Please provide a direct answer based only on the information provided above."
        )

        if self.is_server:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        else:
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    def _generate_llm_response(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        Generate the final LLM response using either the server API or the local model.
        """
        try:
            if self.is_server:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_name,
                    messages=prompt,
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    max_tokens=75,
                )
                answer = response.choices[0].message.content.strip()
            else:
                response = self.llm.generate(
                    [prompt] if isinstance(prompt, str) else [self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )],
                    vllm.SamplingParams(
                        n=1,
                        top_p=0.9,
                        temperature=0.1,
                        skip_special_tokens=True,
                        max_tokens=75,
                    ),
                )[0]
                answer = response.outputs[0].text.strip()

            return answer
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I don't know"

    def _validate_and_clean_answer(self, answer: str, context: str) -> str:
        """
        Validate the answer:
            - If empty or "I don't know", return "I don't know"
            - Remove unnecessary prefixes
            - Check relevance by term overlap
            - Limit length to 75 words
            - Ensure ending punctuation
        """
        if not answer or answer.lower() == "i don't know":
            return "I don't know"

        prefixes_to_remove = [
            "Based on the provided information,",
            "According to the context,",
            "From the information given,",
            "The context indicates that",
            "Based on the available data,"
        ]
        
        cleaned_answer = answer
        for prefix in prefixes_to_remove:
            if cleaned_answer.startswith(prefix):
                cleaned_answer = cleaned_answer[len(prefix):].strip()

        answer_terms = set(cleaned_answer.lower().split())
        context_terms = set(context.lower().split())
        relevance_score = len(answer_terms.intersection(context_terms)) / max(1, len(answer_terms))

        if relevance_score < 0.3:
            return "I don't know"

        words = cleaned_answer.split()
        if len(words) > 75:
            cleaned_answer = ' '.join(words[:75])

        if cleaned_answer and cleaned_answer[-1] not in {'.', '!', '?'}:
            cleaned_answer += '.'

        return cleaned_answer

    def get_batch_size(self) -> int:
        """
        Returns the batch size used for submission.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE
