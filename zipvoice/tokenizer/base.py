# Copyright      2023-2024  Xiaomi Corp.        (authors: Zengwei Yao
#                                                         Han Zhu,
#                                                         Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Tokenizer interface, kept dependency-free (no jieba/piper_phonemize/
pypinyin) so tokenizers that don't need phonemization -- e.g.
MultilingualTokenizer -- don't have to pull those in just to be importable.
"""

from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Abstract base class for tokenizers, defining common interface."""

    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of token id sequences."""
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert list of texts to list of token sequences."""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        """Convert list of token sequences to list of token id sequences."""
        raise NotImplementedError
