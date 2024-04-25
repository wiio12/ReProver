"""Proof search using best-first search.
"""
import os
import sys
import ray
import time
import heapq
import torch
from multilevel_isabelle.src.main.python.pisa_client import (
    IsaDojo, 
    IsabelleError, 
    TacticState, 
    TimeoutError, 
    ProofFinished, 
    ProofGivenUp, 
    DojoInitError, 
    DojoHardTimeoutError, 
    DojoCrashError,
    Theorem,
)

from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool

from common import zip_strict
from prover.search_tree import *
from generator.model import RetrievalAugmentedGenerator, FixedTacticGenerator, DecoderOnlyTacticGenerator


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        rank,
        tac_gen,  # A given tactic generator.
        timeout: int,
        num_sampled_tactics: int,
        history_size: int,
        debug: bool,
    ) -> None:
        self.rank = rank
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.history_size = history_size
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
            

    def search(
        self, repo: dict, thm: Theorem
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.theorem = thm
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        try:
            init_port = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            init_port = int(init_port.split(",")[0])
            init_port = 8005 + init_port
            logger.info(f"Port using: {init_port}, init_port {init_port}, rank {self.rank}, {os.environ.get('CUDA_VISIBLE_DEVICES', 'NONE')}")
            with IsaDojo(
                port=init_port,
                jar_path=repo["jar_path"],
                isa_path=repo["isa_path"],
                working_directory=str(thm.working_directory),
                theory_file_path=str(thm.file_path),
                theorem_name=str(thm.full_name)
            ) as (
                dojo,
                init_state
            ):
                if init_state is not None:
                    self.dojo = dojo
                    self.root = InternalNode(
                        state=init_state,
                        cumulative_logprob=0.0,
                    )
                    self.nodes = {init_state: self.root}
                    self.priority_queue = [self.root]

                    with torch.no_grad():
                        try:
                            self._best_first_search()
                        except DojoCrashError:
                            logger.warning(f"Dojo crashed when proving {thm}")
                            pass
                else:
                    logger.warning(f"IsaDojo fail to init when proving {thm}")
                    self.root = InternalNode(
                        state=init_state,
                        cumulative_logprob=0.0,
                    )
                    self.nodes = {init_state: self.root}

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break
    
    def _get_single_history(self, node):
        history = []
        current_node = node
        while len(current_node.in_edges) > 0:
            edge = current_node.in_edges[0]
            history.append(edge.tactic)
            current_node = edge.src
        history.append(self.root.state.from_tactic)
        history =  list(reversed(history))
        expanded_history = [history[0]]
        for his in history[1:]:
            his = his.split("\n")
            expanded_history.extend(his)
        return expanded_history

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
            from_tactic =  "\n".join(self._get_single_history(search_node)[-self.history_size:])
        else:
            # ts = search_node.state.unsolved_tactic_state
            assert False, "Why this will happen?"
        suggestions = self._generate_tactics(ts, context=from_tactic)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = [
            self._run_tactic(search_node, tactic, logprob)
            for tactic, logprob in suggestions
        ]

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _generate_tactics(self, ts: str, context: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        suggestions = self.tac_gen.generate(
            state=ts,
            context=context,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float) -> Edge:
        def log_result(message, replace_newlines=True):
            if replace_newlines:
                message = message.replace("\n", " ")
            logger.debug(message)
        
        t0 = time.monotonic()
        log_result(f"Running tactic: {tactic}")
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
            if isinstance(response, ProofFinished):
                log_result(f"Result: proof successed! - {str(response.message)}")
            elif isinstance(response, IsabelleError):
                log_result(f"Result: tactic failed! - {str(response)}")
            else:
                log_result(f"Result: duplicate result ! - {str(response.pp)}")
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
                log_result(f'Result: proof successed! - {str(response.message)}')
            elif type(response) in (
                IsabelleError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
                log_result(f'Result: tactic failed! - {str(response)}')
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )
                log_result(f'Result: tactic success! - {response.pp}')

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge

    #########
    # DEBUG #
    #########

    def check_invariants(self):
        """Perform some sanity checks."""
        for node in self.priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                assert node not in self.priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                IsabelleError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                assert node not in self.priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    assert node not in self.priority_queue
                else:
                    assert node in self.priority_queue

                node.check_invariants()


@ray.remote
class CpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a CPU."""

    def __init__(
        self,
        rank,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        use_samping: bool,
        history_size: int,
        debug: bool,
    ) -> None:
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        # else:
        #     tac_gen = RetrievalAugmentedGenerator.load(
        #         ckpt_path, device=torch.device("cpu"), freeze=True
        #     )
        #     if tac_gen.retriever is not None:
        #         if indexed_corpus_path is not None:
        #             tac_gen.retriever.load_corpus(indexed_corpus_path)
        #         tac_gen.retriever.reindex_corpus(batch_size=32)
        else:
            tac_gen = DecoderOnlyTacticGenerator(
                model_name_or_path=ckpt_path, use_sampling=use_samping, device=torch.device("cpu")
            )
        super().__init__(
            rank,
            tac_gen,
            timeout,
            num_sampled_tactics,
            history_size,
            debug,
        )


@ray.remote(num_gpus=1)
class GpuProver(BestFirstSearchProver):
    """Ray actor for running an instance of `BestFirstSearchProver` on a GPU."""

    def __init__(
        self,
        rank: int,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        timeout: int,
        num_sampled_tactics: int,
        use_sampling: bool,
        history_size: int,
        debug: bool,
    ) -> None:
        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        # else:
        #     tac_gen = RetrievalAugmentedGenerator.load(
        #         ckpt_path, device=torch.device("cuda"), freeze=True
        #     )
        #     if tac_gen.retriever is not None:
        #         if indexed_corpus_path is not None:
        #             tac_gen.retriever.load_corpus(indexed_corpus_path)
        #         tac_gen.retriever.reindex_corpus(batch_size=32)
        else:
            tac_gen = DecoderOnlyTacticGenerator(
                model_name_or_path=ckpt_path, use_sampling=use_sampling, device=torch.device("cuda")
            )
        super().__init__(
            rank,
            tac_gen,
            timeout,
            num_sampled_tactics,
            history_size,
            debug,
        )


class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        tactic: Optional[str],
        module: Optional[str],
        num_cpus: int,
        with_gpus: bool,
        timeout: int,
        num_sampled_tactics: int,
        use_sampling: bool,
        history_size: int,
        debug: Optional[bool] = False,
    ) -> None:
        if ckpt_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module
        self.distributed = num_cpus > 1

        if not self.distributed:
            if ckpt_path is None:
                tac_gen = FixedTacticGenerator(tactic, module)
            else:
                device = torch.device("cuda") if with_gpus else torch.device("cpu")
                # tac_gen = RetrievalAugmentedGenerator.load(
                #     ckpt_path, device=device, freeze=True
                # )
                # if tac_gen.retriever is not None:
                #     assert indexed_corpus_path is not None
                #     tac_gen.retriever.load_corpus(indexed_corpus_path)
                tac_gen = DecoderOnlyTacticGenerator(
                    model_name_or_path=ckpt_path, use_sampling=use_sampling, device=device,
                )
            self.prover = BestFirstSearchProver(
                0, tac_gen, timeout, num_sampled_tactics, history_size, debug
            )
            return

        ray.init()
        if with_gpus:
            logger.info(f"Launching {num_cpus} GPU workers.")
            provers = [
                GpuProver.remote(
                    rank,
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    use_sampling=use_sampling,
                    history_size=history_size,
                    debug=debug,
                )
                for rank in range(num_cpus)
            ]
        else:
            logger.info(f"Launching {num_cpus} CPU workers.")
            provers = [
                CpuProver.remote(
                    rank,
                    ckpt_path,
                    indexed_corpus_path,
                    tactic,
                    module,
                    timeout=timeout,
                    num_sampled_tactics=num_sampled_tactics,
                    use_sampling=use_sampling,
                    history_size=history_size,
                    debug=debug,
                )
                for rank in range(num_cpus)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: dict, theorems: List[Theorem]
    ) -> List[SearchResult]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm)
                for thm in theorems
            ]

        try:
            results = list(
                self.prover_pool.map_unordered(
                    lambda p, x: p.search.remote(repo, x),
                    theorems,
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results
