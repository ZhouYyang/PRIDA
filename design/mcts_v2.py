"""Corrected and testable sequence-search implementations for PRIDA.

The legacy implementation intentionally preserves the historical AptaTrans-style
tree search, including its variable-length rollout behavior, so that old and new
searches can be compared under the same reward-evaluation budget.

The corrected implementation makes four changes:
  * four unique 5' -> 3' nucleotide actions (A/C/G/U);
  * every rollout is exactly ``sequence_length`` nucleotides;
  * UCT is used only during search; the committed action is selected by visits
    (default) or mean action value Q;
  * exactly one nucleotide is committed after each search round.

Both implementations depend only on a scorer exposing ``score_many``.  This
keeps the tree-search logic independent of PyTorch and makes it easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


RNA_ALPHABET: Tuple[str, ...] = ("A", "C", "G", "U")
LEGACY_ACTIONS: Tuple[str, ...] = (
    "A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"
)


class SequenceScorer(Protocol):
    """Minimal scorer interface used by all search methods."""

    def score_many(self, sequences: Sequence[str]) -> List[float]:
        ...


class CachedBatchScorer:
    """Batch and optionally cache calls to an arbitrary raw scoring function.

    ``score_requests`` is the algorithmic budget: the number of sequence scores
    requested by a search method. ``model_evaluations`` is the number of
    sequences actually sent through the raw model after cache hits are removed.
    ``forward_batches`` records the number of raw batched forward calls.

    The optional ``group_key`` is useful for language-model inference. Grouping
    sequences by raw length avoids padding-dependent mean pooling and therefore
    reproduces the historical one-sequence-at-a-time representation exactly.
    """

    def __init__(
        self,
        raw_score_many: Callable[[Sequence[str]], Sequence[float]],
        batch_size: int = 32,
        use_cache: bool = True,
        group_key: Optional[Callable[[str], object]] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.raw_score_many = raw_score_many
        self.batch_size = int(batch_size)
        self.use_cache = bool(use_cache)
        self.group_key = group_key
        self.cache: Dict[str, float] = {}
        self.score_requests = 0
        self.model_evaluations = 0
        self.forward_batches = 0

    def _grouped_batches(
        self, indexed_sequences: Sequence[Tuple[int, str]]
    ) -> Iterable[List[Tuple[int, str]]]:
        groups: Dict[object, List[Tuple[int, str]]] = {}
        for item in indexed_sequences:
            key = self.group_key(item[1]) if self.group_key is not None else None
            groups.setdefault(key, []).append(item)
        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                yield group[start : start + self.batch_size]

    def score_many(self, sequences: Sequence[str]) -> List[float]:
        sequences = list(sequences)
        self.score_requests += len(sequences)
        if not sequences:
            return []

        output: List[Optional[float]] = [None] * len(sequences)

        if self.use_cache:
            missing: Dict[str, List[int]] = {}
            for index, sequence in enumerate(sequences):
                if sequence in self.cache:
                    output[index] = self.cache[sequence]
                else:
                    missing.setdefault(sequence, []).append(index)

            indexed_missing = [
                (indices[0], sequence) for sequence, indices in missing.items()
            ]
            for batch in self._grouped_batches(indexed_missing):
                batch_sequences = [sequence for _, sequence in batch]
                values = list(self.raw_score_many(batch_sequences))
                if len(values) != len(batch_sequences):
                    raise RuntimeError("raw_score_many returned the wrong number of scores")
                self.model_evaluations += len(batch_sequences)
                self.forward_batches += 1
                for (_, sequence), value in zip(batch, values):
                    value = float(value)
                    self.cache[sequence] = value
                    for index in missing[sequence]:
                        output[index] = value
        else:
            indexed = list(enumerate(sequences))
            for batch in self._grouped_batches(indexed):
                batch_sequences = [sequence for _, sequence in batch]
                values = list(self.raw_score_many(batch_sequences))
                if len(values) != len(batch_sequences):
                    raise RuntimeError("raw_score_many returned the wrong number of scores")
                self.model_evaluations += len(batch_sequences)
                self.forward_batches += 1
                for (index, _), value in zip(batch, values):
                    output[index] = float(value)

        if any(value is None for value in output):
            raise RuntimeError("one or more requested sequences were not scored")
        return [float(value) for value in output]  # type: ignore[arg-type]


@dataclass
class SearchResult:
    method: str
    candidate: str
    score_requests: int
    evaluated: List[Tuple[str, float]]
    best_evaluated_sequence: str
    best_evaluated_score: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class _TreeNode:
    sequence: str
    parent: Optional["_TreeNode"] = None
    action: Optional[str] = None
    actions: Tuple[str, ...] = RNA_ALPHABET
    children: Dict[str, "_TreeNode"] = field(default_factory=dict)
    untried_actions: List[str] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = list(self.actions)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def effective_visits(self) -> int:
        return self.visits + self.virtual_visits


class CorrectedMCTS:
    """Four-action, fixed-length, batched MCTS with one-base commitment."""

    def __init__(
        self,
        scorer: SequenceScorer,
        sequence_length: int = 20,
        exploration_constant: float = 1.0 / math.sqrt(2.0),
        final_policy: str = "visits",
        batch_size: int = 32,
        seed: int = 42,
        alphabet: Tuple[str, ...] = RNA_ALPHABET,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if final_policy not in {"visits", "q"}:
            raise ValueError("final_policy must be either 'visits' or 'q'")
        self.scorer = scorer
        self.sequence_length = int(sequence_length)
        self.exploration_constant = float(exploration_constant)
        self.final_policy = final_policy
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.alphabet = tuple(alphabet)
        self.rng = random.Random(self.seed)

    def _uct(self, parent: _TreeNode, child: _TreeNode) -> float:
        child_visits = child.effective_visits
        if child_visits == 0:
            return float("inf")
        parent_visits = max(1, parent.effective_visits)
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_visits + 1.0) / child_visits
        )
        return child.q_value + exploration

    def _choose_uct_child(self, node: _TreeNode) -> _TreeNode:
        values = [(self._uct(node, child), child) for child in node.children.values()]
        best_value = max(value for value, _ in values)
        tied = [child for value, child in values if value == best_value]
        return self.rng.choice(tied)

    def _select_expand_rollout(
        self, root: _TreeNode
    ) -> Tuple[List[_TreeNode], str]:
        node = root
        path = [node]

        while len(node.sequence) < self.sequence_length:
            if node.untried_actions:
                index = self.rng.randrange(len(node.untried_actions))
                action = node.untried_actions.pop(index)
                child = _TreeNode(
                    sequence=node.sequence + action,
                    parent=node,
                    action=action,
                    actions=self.alphabet,
                )
                node.children[action] = child
                node = child
                path.append(node)
                break
            node = self._choose_uct_child(node)
            path.append(node)

        rollout = node.sequence
        while len(rollout) < self.sequence_length:
            rollout += self.rng.choice(self.alphabet)

        # Reserve the path while the rest of the batch is selected. This small
        # virtual-visit mechanism prevents all members of a batch from selecting
        # exactly the same branch before their rewards are available.
        for path_node in path:
            path_node.virtual_visits += 1
        return path, rollout

    @staticmethod
    def _finish_simulation(path: Sequence[_TreeNode], score: float) -> None:
        for node in path:
            node.virtual_visits -= 1
            node.visits += 1
            node.value_sum += float(score)

    def _commit_child(self, root: _TreeNode) -> _TreeNode:
        visited = [child for child in root.children.values() if child.visits > 0]
        if not visited:
            raise RuntimeError("cannot commit an action before any child is evaluated")

        if self.final_policy == "visits":
            primary = max(child.visits for child in visited)
            candidates = [child for child in visited if child.visits == primary]
            secondary = max(child.q_value for child in candidates)
            candidates = [child for child in candidates if child.q_value == secondary]
        else:
            primary = max(child.q_value for child in visited)
            candidates = [child for child in visited if child.q_value == primary]
            secondary = max(child.visits for child in candidates)
            candidates = [child for child in candidates if child.visits == secondary]
        return self.rng.choice(candidates)

    def search(self, total_budget: int) -> SearchResult:
        if total_budget < self.sequence_length:
            raise ValueError(
                "total_budget must be at least sequence_length so every move "
                "receives at least one simulation"
            )

        per_move = [total_budget // self.sequence_length] * self.sequence_length
        for index in range(total_budget % self.sequence_length):
            per_move[index] += 1

        prefix = ""
        evaluated: List[Tuple[str, float]] = []
        move_metadata: List[Dict[str, object]] = []

        for move_index, move_budget in enumerate(per_move):
            root = _TreeNode(sequence=prefix, actions=self.alphabet)
            completed = 0
            while completed < move_budget:
                current_batch = min(self.batch_size, move_budget - completed)
                selections = [
                    self._select_expand_rollout(root) for _ in range(current_batch)
                ]
                sequences = [rollout for _, rollout in selections]
                scores = self.scorer.score_many(sequences)
                for (path, sequence), score in zip(selections, scores):
                    self._finish_simulation(path, score)
                    evaluated.append((sequence, float(score)))
                completed += current_batch

            chosen = self._commit_child(root)
            prefix = chosen.sequence
            move_metadata.append(
                {
                    "move": move_index + 1,
                    "committed_base": chosen.action,
                    "prefix": prefix,
                    "visits": chosen.visits,
                    "q_value": chosen.q_value,
                    "simulations": move_budget,
                }
            )

        best_sequence, best_score = max(evaluated, key=lambda item: item[1])
        return SearchResult(
            method="corrected_mcts",
            candidate=prefix,
            score_requests=len(evaluated),
            evaluated=evaluated,
            best_evaluated_sequence=best_sequence,
            best_evaluated_score=float(best_score),
            metadata={
                "sequence_length": self.sequence_length,
                "alphabet": list(self.alphabet),
                "exploration_constant": self.exploration_constant,
                "final_policy": self.final_policy,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "per_move": move_metadata,
            },
        )


class RandomSearch:
    """Uniform random search with the same sequence-evaluation budget."""

    def __init__(
        self,
        scorer: SequenceScorer,
        sequence_length: int = 20,
        batch_size: int = 32,
        seed: int = 42,
        alphabet: Tuple[str, ...] = RNA_ALPHABET,
    ) -> None:
        self.scorer = scorer
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.alphabet = tuple(alphabet)
        self.rng = random.Random(self.seed)

    def search(self, total_budget: int) -> SearchResult:
        if total_budget < 1:
            raise ValueError("total_budget must be positive")
        if total_budget > len(self.alphabet) ** self.sequence_length:
            raise ValueError("budget exceeds the number of possible unique sequences")

        sequences: List[str] = []
        seen = set()
        while len(sequences) < total_budget:
            sequence = "".join(
                self.rng.choice(self.alphabet) for _ in range(self.sequence_length)
            )
            if sequence not in seen:
                seen.add(sequence)
                sequences.append(sequence)

        evaluated: List[Tuple[str, float]] = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            values = self.scorer.score_many(batch)
            evaluated.extend((sequence, float(value)) for sequence, value in zip(batch, values))

        best_sequence, best_score = max(evaluated, key=lambda item: item[1])
        return SearchResult(
            method="random_search",
            candidate=best_sequence,
            score_requests=len(evaluated),
            evaluated=evaluated,
            best_evaluated_sequence=best_sequence,
            best_evaluated_score=float(best_score),
            metadata={
                "sequence_length": self.sequence_length,
                "alphabet": list(self.alphabet),
                "batch_size": self.batch_size,
                "seed": self.seed,
            },
        )


@dataclass
class _LegacyNode:
    token: str = ""
    parent: Optional["_LegacyNode"] = None
    is_root: bool = False
    depth: int = 0
    visits: int = 1
    value_sum: float = 0.0
    children: List[Optional["_LegacyNode"]] = field(
        default_factory=lambda: [None] * len(LEGACY_ACTIONS)
    )

    def backpropagate(self, score: float) -> None:
        self.visits += 1
        if self.is_root:
            return
        self.value_sum += float(score)
        if self.parent is not None:
            self.parent.backpropagate(score)

    @property
    def uct(self) -> float:
        if self.parent is None:
            return self.value_sum / self.visits
        return (self.value_sum / self.visits) + math.sqrt(
            math.log(self.parent.visits) / (2.0 * self.visits)
        )


class LegacyMCTS:
    """Scorer-agnostic snapshot of the historical eight-action MCTS.

    This class intentionally retains the historical rollout-length calculation
    so that comparisons measure the actual old behavior rather than a partially
    corrected approximation.
    """

    def __init__(
        self,
        scorer: SequenceScorer,
        sequence_length: int = 20,
        iteration: int = 50,
        seed: int = 42,
    ) -> None:
        self.scorer = scorer
        self.sequence_length = int(sequence_length)
        self.iteration = int(iteration)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.base = ""
        self.root = _LegacyNode(is_root=True, depth=0)
        self.evaluated: List[Tuple[str, float]] = []

    @staticmethod
    def reconstruct(encoded: str) -> str:
        sequence = ""
        for index in range(0, len(encoded), 2):
            if encoded[index] == "_":
                sequence = sequence + encoded[index + 1]
            else:
                sequence = encoded[index] + sequence
        return sequence

    def _simulate(self, node: _LegacyNode) -> float:
        tokens: List[str] = []
        current = node
        while not current.is_root:
            tokens.append(current.token)
            if current.parent is None:
                break
            current = current.parent
        encoded = self.base + "".join(reversed(tokens))

        # Intentionally historical: the loop counts encoded characters, but
        # appends a two-character action each time. At depth 1 with L=20 this
        # evaluates a 39-nt rollout rather than a 20-nt rollout.
        for _ in range((self.sequence_length * 2) - len(encoded)):
            encoded += self.rng.choice(LEGACY_ACTIONS)
        sequence = self.reconstruct(encoded)
        score = float(self.scorer.score_many([sequence])[0])
        self.evaluated.append((sequence, score))
        return score

    def _select(self, node: _LegacyNode) -> _LegacyNode:
        if node.depth == self.sequence_length:
            return self.root

        unexpanded = [index for index, child in enumerate(node.children) if child is None]
        if unexpanded:
            # The historical code samples from all actions and descends when it
            # hits an existing child. Repeating that behavior exactly is
            # equivalent to choosing one random action here.
            action_index = self.rng.randrange(len(LEGACY_ACTIONS))
            existing = node.children[action_index]
            if existing is not None:
                return existing
            child = _LegacyNode(
                token=LEGACY_ACTIONS[action_index],
                parent=node,
                depth=node.depth + 1,
            )
            node.children[action_index] = child
            score = self._simulate(child)
            child.backpropagate(score)
            return self.root

        children = [child for child in node.children if child is not None]
        return max(children, key=lambda child: child.uct)  # type: ignore[arg-type]

    def _best_subsequence(self) -> str:
        node = self.root
        encoded = self.base
        remaining_encoded_characters = (self.sequence_length * 2) - len(encoded)
        for _ in range(remaining_encoded_characters):
            children = [child for child in node.children if child is not None]
            if not children:
                break
            node = max(children, key=lambda child: child.uct)
            encoded += node.token
        return encoded

    def search(self) -> SearchResult:
        while len(self.base) < self.sequence_length * 2:
            previous_length = len(self.base)
            node = self.root
            for _ in range(self.iteration):
                node = self._select(node)
            self.base = self._best_subsequence()
            if len(self.base) == previous_length:
                raise RuntimeError(
                    "legacy search made no progress; increase the iteration count"
                )
            self.root = _LegacyNode(
                is_root=True, depth=len(self.base) // 2
            )

        candidate = self.reconstruct(self.base)
        best_sequence, best_score = max(self.evaluated, key=lambda item: item[1])
        return SearchResult(
            method="legacy_mcts",
            candidate=candidate,
            score_requests=len(self.evaluated),
            evaluated=self.evaluated,
            best_evaluated_sequence=best_sequence,
            best_evaluated_score=float(best_score),
            metadata={
                "sequence_length": self.sequence_length,
                "actions": list(LEGACY_ACTIONS),
                "iteration": self.iteration,
                "seed": self.seed,
                "historical_rollout_length_bug": True,
            },
        )
