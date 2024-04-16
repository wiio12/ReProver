"""Proof search using best-first search.
"""
import os
from queue import Queue
import random
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
from prover.search_tree_multilevel import *
from generator.model import RetrievalAugmentedGenerator, FixedTacticGenerator, DecoderOnlyTacticGenerator, DummyTacticGenerator
from itertools import chain


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
        debug: bool,
    ) -> None:
        self.rank = rank
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG", format=r"<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <red>{extra}</red> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | - <level>{message}</level>")
            

    def search(
        self, repo: dict, thm: Theorem
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.theorem = thm
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        try:
            with IsaDojo(
                port=8000+int(self.rank),
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
                        is_root=True
                    )
                    self.nodes = {init_state: self.root}
                    self.priority_queue = [self.root]

                    with torch.no_grad():
                        try:
                            with logger.contextualize(root_level=0):
                                self._best_first_search(self.root, self.nodes, self.priority_queue, root_level=0)
                        except DojoCrashError:
                            logger.warning(f"Dojo crashed when proving {thm}")
                            pass
                else:
                    logger.warning(f"IsaDojo fail to init when proving {thm}")
                    self.root = InternalNode(
                        state=init_state,
                        cumulative_logprob=0.0,
                        is_root=True,
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

    def _best_first_search(self, root, nodes, priority_queue, root_level) -> None:
        time_start = time.monotonic()

        while True:
            if len(priority_queue) == 0 or root_level > 10:
                logger.info(f"Ran out of nodes to search or root level {root_level} too deep")
                root.status = Status.FAILED
                root.boardcast_failure()
                for edge in root.in_edges:
                    edge.src._recompute_status()
                break

            try:
                self._step(root, nodes, priority_queue)
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if root.status == Status.PROVED:
                    logger.info(f"Root {root} found a proof but timed out.")
                root.status = Status.OPEN
                logger.info(f"Root {root} search timed out.")
                break

            if root.status == Status.FAILED:
                logger.info(f"Root {root} failed early!")
                break

            if root.status == Status.PROVED:
                logger.info(f"Root {root} Found a proof!!")
                break

            if root.status == Status.HALF_PROVED:
                # last first
                c_root, succes_trajs = self._get_last_half_proved_roots(root)
                for traj in list(chain.from_iterable(succes_trajs)):
                    logger.debug(traj)
                c_nodes = {c_root.state: c_root}
                c_priority_queue = [c_root]
                logger.info(f">>> Root {root} found a proof, going deeper with root {c_root}")
                with logger.contextualize(root_level=root_level+1):
                    self._best_first_search(c_root, c_nodes, c_priority_queue, root_level=root_level+1)

    def _get_last_half_proved_roots(self, root):
        sorry_roots = []
        success_node = []
        node_queue = Queue()
        node_queue.put(root)
        while not node_queue.empty():
            node = node_queue.get()
            if isinstance(node, ProofFinishedNode):
                success_node.append(node)
            if not isinstance(node, InternalNode):
                continue
            for edge in node.out_edges:
                if edge.dst.status in [Status.PROVED, Status.HALF_PROVED]:
                    print(edge)
                    node_queue.put(edge.dst)
                    if isinstance(edge, SorryEdge) and edge.sorry_root.status == Status.OPEN:
                        sorry_roots.append(edge.sorry_root)
        success_node = success_node[:10]
        success_traj = [self._get_traj(node, status_requirements = [Status.PROVED, Status.HALF_PROVED]) for node in success_node]
        return sorry_roots[-1], success_traj

    def _get_traj(self, node, status_requirements=None):
        if len(node.in_edges) == 0:
            return [Trajectory([node])]
        
        all_traj = []
        for edge in node.in_edges:
            if status_requirements and edge.src.status not in status_requirements:
                continue
            trajs_to_node = self._get_traj(edge.src, status_requirements=status_requirements)
            for traj in trajs_to_node:
                all_traj.append(traj + Trajectory([edge, node]))
        # all_traj = random.sample(all_traj, min(3, len(all_traj)))
        return all_traj

    def _step(self, root, nodes, priority_queue):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(priority_queue)
        logger.debug(f"Expanding node: {search_node}")
        logger.debug(f"Privous Step: {self._get_traj(search_node)[0]}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
            from_tactic = search_node.state.from_tactic
        else:
            # ts = search_node.state.unsolved_tactic_state
            assert False, "Why this will happen?"
        suggestions = self._generate_tactics(ts, context=from_tactic)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = [
            self._run_tactic(search_node, tactic, logprob, root=root, nodes=nodes, priority_queue=priority_queue)
            for tactic, logprob in suggestions
        ]

        results = list(filter(lambda x: x is not None, results))

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        # if self.debug:
        #     assert self.num_expansions == sum(
        #         node.is_explored
        #         for node in self.nodes.values()
        #         if isinstance(node, InternalNode)
        #     )
        #     self.check_invariants()

    def _generate_tactics(self, ts: str, context: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        suggestions = self.tac_gen.generate(
            state=ts,
            context=context,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        # change the order of the tactics, make tactic with sorry come higher:
        def custom_sort_key(item):
            contains_sorry = 'sorry' in item[0].lower()
            return (not contains_sorry, item[1])

        suggestions = sorted(suggestions, key=custom_sort_key)


        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(self, node: InternalNode, tactic: str, logprob: float, root: InternalNode, nodes, priority_queue) -> Edge:
        def log_result(message, replace_newlines=True):
            if replace_newlines:
                message = message.replace("\n", " ")
            logger.debug(message)
        
        t0 = time.monotonic()
        log_result(f"Running tactic: {tactic}")
        response = self.dojo.run_tac(node.state, tactic, root.state)
        no_sorry_response = None
        sorry_root = None
        if isinstance(response, TacticState) and "sorry" in tactic:
            no_sorry_tactic = tactic[:tactic.index("sorry")].strip()
            no_sorry_response = self.dojo.run_tac(node.state, no_sorry_tactic, root.state)
            if isinstance(no_sorry_response, IsabelleError):
                response = no_sorry_response

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            # TODO: I don't know if this is good to do
            result_node = nodes[response]

            # if the result is itself, this have no meaning!
            # if tactic == "thus?thesis sorry":
            #     print("jherere")
            trajs = self._get_traj(node)
            if any([result_node in traj for traj in trajs]):
                return None

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
                logger.info("hererererer!!!SSSS!!!!!!!!!!!!!!!")
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
                if no_sorry_response is not None:
                    assert isinstance(no_sorry_response, TacticState)
                    sorry_root = InternalNode(
                        state=no_sorry_response,
                        cumulative_logprob=0,    # TODO: correct way to calculate the logprob, I think for now it's 0 is good
                        is_root=True
                    )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(priority_queue, result_node)  # type: ignore
            
            # if sorry_root.status == Status.OPEN:
            #     heapq.heappush(priority_queue, sorry_root)
                
        if sorry_root is None and "sorry" in tactic and isinstance(response, TacticState):
            assert response in nodes
            if no_sorry_response in nodes:
                flag = False
                for edge in result_node.in_edges:
                    if isinstance(edge, SorryEdge):
                        if edge.sorry_root == nodes[no_sorry_response]:
                            flag = True
                if not flag:
                    logger.warning("There are common sorry root but diffent result after sorry root in this tree")
                assert isinstance(no_sorry_response, TacticState)
                sorry_root = nodes[no_sorry_response]
                assert sorry_root.is_root is True
            if sorry_root is None:
                assert isinstance(no_sorry_response, TacticState)
                sorry_root = InternalNode(
                    state=no_sorry_response,
                    cumulative_logprob=0,    # TODO: correct way to calculate the logprob, I think for now it's 0 is good
                    is_root=True
                )

        
        # if the tactic gives a root node, don's search here, return None
        if isinstance(result_node, InternalNode) and result_node.is_root:
            return None
        


        # Record the new node and add it to the search queue.
        nodes[response] = result_node
        if sorry_root is not None:
            nodes[no_sorry_response] = sorry_root

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        if sorry_root is None:
            edge = Edge(tactic=tactic, src=node, dst=result_node)
        else:
            edge = SorryEdge(tactic=tactic, src=node, dst=result_node, sorry_root=sorry_root, sorry_tactic=no_sorry_tactic)

        # if isinstance(result_node, InternalNode):
        result_node.in_edges.append(edge)
        
        if sorry_root is not None:
            sorry_root.in_edges.append(edge)

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
        debug: bool,
    ) -> None:
        if ckpt_path is None:
            tac_gen = DummyTacticGenerator(
                    data_path="/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/multilevel/test.json"
            )
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
                model_name_or_path=ckpt_path, device=torch.device("cpu")
            )
        super().__init__(
            rank,
            tac_gen,
            timeout,
            num_sampled_tactics,
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
        debug: bool,
    ) -> None:
        if ckpt_path is None:
            tac_gen = DummyTacticGenerator(
                    data_path="/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/multilevel/test.json"
            )
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
                model_name_or_path=ckpt_path, device=torch.device("cuda")
            )
        super().__init__(
            rank,
            tac_gen,
            timeout,
            num_sampled_tactics,
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
        debug: Optional[bool] = False,
    ) -> None:
        if ckpt_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module
        self.distributed = num_cpus > 1

        if not self.distributed:
            if ckpt_path is None:
                tac_gen = DummyTacticGenerator(
                    data_path="/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/multilevel/test.json"
                )
            else:
                device = torch.device("cuda") if with_gpus else torch.device("cpu")
                tac_gen = DecoderOnlyTacticGenerator(
                    model_name_or_path=ckpt_path, device=device,
                )

                
            self.prover = BestFirstSearchProver(
                0, tac_gen, timeout, num_sampled_tactics, debug
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
