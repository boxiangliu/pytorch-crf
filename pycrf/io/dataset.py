"""Defines dataset class."""

from collections.abc import Sized, Iterable
import random
from typing import List, Tuple, Generator

import torch

from .vocab import Vocab, SourceType, TargetType


GeneratorType = Generator[Tuple[SourceType, TargetType], None, None]
GeneratorType2 = Generator[Tuple[SourceType, TargetType, List[str], List[str]], None, None]


class Dataset(Sized, Iterable):
    """Class for abstracting training and testing datasets."""

    def __init__(self, is_test: bool = False) -> None:
        self.source: List[SourceType] = []
        self.target: List[TargetType] = []
        self.is_test = is_test

    def __getitem__(self, key: int) -> Tuple[SourceType, TargetType]:
        return self.source[key], self.target[key]

    def __iter__(self) -> GeneratorType:
        for src, tgt in zip(self.source, self.target):
            yield src, tgt

    def __len__(self) -> int:
        return len(self.source)

    def __bool__(self) -> bool:
        return len(self.source) > 0

    def append(self,
               src: List[str],
               tgt: List[str],
               vocab: Vocab,
               device: torch.device = None,
               sent_context: str = None) -> None:
        """Append a new training example."""
        src_tensor = vocab.sent2tensor(src, device=device,
                                       sent_context=sent_context,
                                       test=self.is_test)
        tgt_tensor = vocab.labs2tensor(tgt, device=device,
                                       test=self.is_test)
        self.source.append(src_tensor)
        self.target.append(tgt_tensor)

    def shuffle(self) -> None:
        """Shuffle source and targets together."""
        combined = list(zip(self.source, self.target))
        random.shuffle(combined)
        self.source[:], self.target[:] = zip(*combined)

    @staticmethod
    def read_file(fname: str,
                  vocab: Vocab,
                  device: torch.device = None,
                  sent_context: str = None,
                  test: bool = True) -> GeneratorType2:
        """
        Read sentences from a file.

        Parameters
        ----------
        fname : str
            The path to the file to load. Files are assumed to look like this:

            ::

                Hi     O
                there  O

                how    O
                are    O
                you    O
                ?      O

            Each sentence is followed by an empty line, and each line
            corresponding to a token in the sentence begins with the token,
            then a tab character, then the corresponding label.

        vocab : pycrf.io.Vocab
            The vocab instance to apply to the sentences.

        device : torch.device, optional
            The device to send the tensors to.

        sent_context : str, optional
            The sentence-level context.

        test : bool, optional
            Whether this represents a new test sentence.

        Yields
        ------
        Tuple[SourceType, TargetType, List[str], List[str]]

        """
        with open(fname, "r") as datafile:
            breakpoint()
            src: List[str] = []
            tgt: List[str] = []
            for line in datafile.readlines():
                line_list = line.rstrip().split('\t')
                if len(line_list) == 1:  # end of sentence.
                    target_tensor = vocab.labs2tensor(tgt, device=device,
                                                      test=test)
                    source_tensor = vocab.sent2tensor(src, device=device,
                                                      sent_context=sent_context,
                                                      test=test)
                    yield source_tensor, target_tensor, src, tgt
                    src = []
                    tgt = []
                else:
                    src.append(line_list[0])
                    tgt.append(line_list[1])

    def load_file(self,
                  fname: str,
                  vocab: Vocab,
                  limit: int = None,
                  device: torch.device = None,
                  sent_context: str = None) -> None:
        """
        Load sentences from a file.

        Parameters
        ----------
        fname : str
            The path to the file to load. Files are assumed to look like this:

            ::

                Hi     O
                there  O

                how    O
                are    O
                you    O
                ?      O

            Each sentence is followed by an empty line, and each line
            corresponding to a token in the sentence begins with the token,
            then a tab character, then the corresponding label.

        vocab : pycrf.io.Vocab
            The vocab instance to apply to the sentences.

        limit : int, optional
            If set, will only load this many examples.

        device : torch.device, optional
            The device to send the tensors to.

        sent_context : str, optional
            The sentence-level context.

        Returns
        -------
        None

        """
        print("Loading file {:s}".format(fname), flush=True)
        i = 0
        cursor = self.read_file(fname, vocab, device=device,
                                sent_context=sent_context,
                                test=self.is_test)
        for source, target, _, _ in cursor:
            self.source.append(source)
            self.target.append(target)
            i += 1
            if limit is not None and i == limit:
                break
