"""Retrieval laboratory for Lost Melbourne."""

from .chunking import build_passages
from .corpus import CorpusValidationError, HistoricalCorpus
from .dense import DenseIndex, HashingEncoder, SentenceTransformerEncoder
from .hybrid import reciprocal_rank_fusion
from .models import RetrievalResult, SearchPassage
from .query import QueryTransformer
from .sparse import BM25Index, TfidfIndex
from .structured import StructuredRetriever

__all__ = [
    "BM25Index", "CorpusValidationError", "DenseIndex", "HashingEncoder", "HistoricalCorpus",
    "QueryTransformer", "RetrievalResult", "SearchPassage", "SentenceTransformerEncoder",
    "StructuredRetriever", "TfidfIndex", "build_passages", "reciprocal_rank_fusion",
]
