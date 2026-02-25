"""
Chess.com API → PGN parsing → Tokenized Dataset Pipeline
This is the pipeline to extract chess games from the Chess.com API, parse the PGN format to get clean move lists, build a move-level vocabulary, and create a PyTorch Dataset for next-move prediction.
Goal: next-move prediction using algebraic notation tokens
"""

import re
import json
import time
import requests
import pickle
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. DATA COLLECTION
# =============================================================================

HEADERS = {"User-Agent": "chess-transformer/1.0 your@email.com"}


def get_all_archives(username: str) -> list[str]:
    """Return list of monthly archive URLs for a user."""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get("archives", [])


def fetch_games_from_archive(archive_url: str) -> list[dict]:
    """Fetch all games from a single monthly archive URL."""
    resp = requests.get(archive_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get("games", [])


def fetch_recent_games(
    username: str,
    n_months: int = 3,
    time_control: Optional[str] = None,
) -> list[dict]:
    """
    Fetch games from the most recent n_months archives.

    Args:
        username:     Chess.com username
        n_months:     How many months back to fetch (most recent first)
        time_control: Optional filter e.g. "180" for 3-min blitz, "600" for 10-min rapid
    """
    archives = get_all_archives(username)
    recent_archives = archives[-n_months:][::-1]  # most recent first

    all_games = []
    for archive_url in recent_archives:
        print(f"Fetching: {archive_url}")
        games = fetch_games_from_archive(archive_url)
        if time_control:
            games = [g for g in games if g.get("time_control") == time_control]
        all_games.extend(games)
        time.sleep(0.5)  # be polite to the API

    print(f"Fetched {len(all_games)} games total.")
    return all_games


def save_raw_games(games: list[dict], path: str | Path = "raw_games.json"):
    with open(path, "w") as f:
        json.dump(games, f)
    print(f"Saved {len(games)} raw games to {path}")


def load_raw_games(path: str | Path = "raw_games.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


# =============================================================================
# 2. PGN PARSING
# =============================================================================

def parse_pgn_headers(pgn: str) -> dict:
    """Extract key-value pairs from PGN header brackets."""
    headers = {}
    for line in pgn.strip().splitlines():
        m = re.match(r'\[(\w+)\s+"(.*)"\]', line)
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def extract_moves(pgn: str) -> list[str]:
    """
    Strip PGN annotations and return a clean list of SAN move strings.

    e.g. "1. e4 {[%clk 0:02:59]} 1... c6" -> ["e4", "c6"]
    """
    # Split off headers (everything before the blank line)
    parts = pgn.strip().split("\n\n", 1)
    if len(parts) < 2:
        return []
    movetext = parts[1]

    # Remove comments in curly braces (clock annotations, engine evals, etc.)
    movetext = re.sub(r'\{[^}]*\}', '', movetext)

    # Remove NAG symbols ($1, $2, ...)
    movetext = re.sub(r'\$\d+', '', movetext)

    # Remove variations in parentheses (repeat until fully stripped for nested cases)
    prev = None
    while prev != movetext:
        prev = movetext
        movetext = re.sub(r'\([^)]*\)', '', movetext)

    # Remove move numbers: "1." "1..." "34."
    movetext = re.sub(r'\d+\.+', '', movetext)

    # Remove result tokens
    movetext = re.sub(r'(1-0|0-1|1/2-1/2|\*)', '', movetext)

    # Split and clean
    moves = [m.strip() for m in movetext.split() if m.strip()]
    return moves


def parse_game(game_dict: dict) -> Optional[dict]:
    """
    Parse a single raw game dict from the Chess.com API.
    Returns a structured dict with headers + move list, or None if unparseable.
    """
    pgn = game_dict.get("pgn", "")
    if not pgn:
        return None

    moves = extract_moves(pgn)
    if len(moves) < 5:  # skip extremely short/aborted games
        return None

    headers = parse_pgn_headers(pgn)

    return {
        "moves": moves,
        "white": headers.get("White"),
        "black": headers.get("Black"),
        "result": headers.get("Result"),
        "white_elo": headers.get("WhiteElo"),
        "black_elo": headers.get("BlackElo"),
        "time_control": headers.get("TimeControl"),
        "eco": headers.get("ECO"),
        "termination": headers.get("Termination"),
        "url": game_dict.get("url"),
    }


def parse_all_games(raw_games: list[dict]) -> list[dict]:
    parsed = [parse_game(g) for g in raw_games]
    parsed = [g for g in parsed if g is not None]
    print(f"Successfully parsed {len(parsed)} / {len(raw_games)} games.")
    return parsed


# =============================================================================
# 3. VOCABULARY
# =============================================================================

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class ChessVocab:
    """
    Builds and stores a move-level vocabulary from a corpus of parsed games.

    Each unique SAN token (e4, Nf3, O-O, Qxf6+, etc.) gets an integer ID.
    Special tokens: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
    """

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def build(self, games: list[dict], min_freq: int = 1):
        """
        Build vocab from a list of parsed game dicts.

        Args:
            games:    list of dicts with a "moves" key
            min_freq: minimum frequency to include a move token
                      (rarely needed for chess since vocab is closed)
        """
        counts = Counter(m for g in games for m in g["moves"])
        move_tokens = sorted(t for t, c in counts.items() if c >= min_freq)

        all_tokens = SPECIAL_TOKENS + move_tokens
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        print(f"Vocabulary size: {len(self.token_to_id)} tokens")
        print(f"  ({len(SPECIAL_TOKENS)} special + {len(move_tokens)} move tokens)")

    def encode(self, moves: list[str]) -> list[int]:
        return [self.token_to_id.get(m, UNK_ID) for m in moves]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, "<UNK>") for i in ids]

    def __len__(self):
        return len(self.token_to_id)

    def save(self, path: str | Path = "chess_vocab.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved vocab to {path}")

    @staticmethod
    def load(path: str | Path = "chess_vocab.pkl") -> "ChessVocab":
        with open(path, "rb") as f:
            return pickle.load(f)


# =============================================================================
# 4. DATASET
# =============================================================================

class ChessNextMoveDataset(Dataset):
    """
    PyTorch Dataset for next-move prediction.

    Each sample is:
        input:  [SOS, m1, m2, ..., mN]       (length: seq_len)
        target: [m1,  m2, ..., mN, EOS]      (length: seq_len)

    i.e. the target is the input shifted left by one — standard causal LM setup.
    The model learns P(m_{t+1} | m_1, ..., m_t) for all t simultaneously.

    Sequences shorter than max_len are right-padded with <PAD>.
    Sequences longer than max_len are truncated (rare in chess).
    """

    def __init__(
        self,
        games: list[dict],
        vocab: ChessVocab,
        max_len: int = 128,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = self._build_samples(games)

    def _build_samples(self, games: list[dict]) -> list[dict]:
        samples = []
        for game in games:
            move_ids = self.vocab.encode(game["moves"])

            # Truncate if needed (most chess games are 30-80 moves = 60-160 tokens)
            move_ids = move_ids[:self.max_len - 1]  # leave room for SOS/EOS

            # input:  SOS + moves (right-padded to max_len)
            # target: moves + EOS (right-padded to max_len)
            inp = [SOS_ID] + move_ids
            tgt = move_ids + [EOS_ID]

            # Pad to max_len
            pad_len = self.max_len - len(inp)
            inp = inp + [PAD_ID] * pad_len
            tgt = tgt + [PAD_ID] * pad_len

            samples.append({
                "input_ids":  torch.tensor(inp, dtype=torch.long),
                "target_ids": torch.tensor(tgt, dtype=torch.long),
                # attention mask: 1 for real tokens, 0 for padding
                "attention_mask": torch.tensor(
                    [1] * (self.max_len - pad_len) + [0] * pad_len,
                    dtype=torch.long
                ),
            })

        print(f"Built {len(samples)} training samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# 5. FULL PIPELINE (put it all together)
# =============================================================================

def build_pipeline(
    username: str,
    n_months: int = 6,
    time_control: Optional[str] = None,
    max_len: int = 128,
    batch_size: int = 32,
    save_dir: str = "chess_data",
) -> tuple[DataLoader, ChessVocab]:
    """
    End-to-end: Chess.com API → tokenized DataLoader + vocab.

    Returns:
        dataloader: ready to iterate for training
        vocab:      ChessVocab (needed to decode model outputs)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    raw_path    = save_dir / "raw_games.json"
    parsed_path = save_dir / "parsed_games.json"
    vocab_path  = save_dir / "chess_vocab.pkl"

    # --- Step 1: Fetch ---
    if raw_path.exists():
        print(f"Loading cached raw games from {raw_path}")
        raw_games = load_raw_games(raw_path)
    else:
        raw_games = fetch_recent_games(username, n_months=n_months, time_control=time_control)
        save_raw_games(raw_games, raw_path)

    # --- Step 2: Parse ---
    if parsed_path.exists():
        print(f"Loading cached parsed games from {parsed_path}")
        with open(parsed_path) as f:
            parsed_games = json.load(f)
    else:
        parsed_games = parse_all_games(raw_games)
        with open(parsed_path, "w") as f:
            json.dump(parsed_games, f, indent=2)
        print(f"Saved parsed games to {parsed_path}")

    # --- Step 3: Vocabulary ---
    if vocab_path.exists():
        print(f"Loading cached vocab from {vocab_path}")
        vocab = ChessVocab.load(vocab_path)
    else:
        vocab = ChessVocab()
        vocab.build(parsed_games)
        vocab.save(vocab_path)

    # --- Step 4: Dataset + DataLoader ---
    dataset = ChessNextMoveDataset(parsed_games, vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, vocab


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    USERNAME = "switz2000"

    dataloader, vocab = build_pipeline(
        username=USERNAME,
        n_months=1,          # fetch last 6 months of games
        time_control="180",  # blitz only; set None for all time controls
        max_len=128,
        batch_size=32,
    )

    # Inspect a batch
    batch = next(iter(dataloader))
    print("input_ids shape:     ", batch["input_ids"].shape)      # (32, 128)
    print("target_ids shape:    ", batch["target_ids"].shape)     # (32, 128)
    print("attention_mask shape:", batch["attention_mask"].shape) # (32, 128)
    print("Vocab size:", len(vocab))

    # Decode a sample to sanity-check
    sample_input  = vocab.decode(batch["input_ids"][0].tolist())
    sample_target = vocab.decode(batch["target_ids"][0].tolist())
    print("\nSample input: ", sample_input[:10])
    print("Sample target:", sample_target[:10])