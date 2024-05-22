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

from prover.search_tree_multilevel import *
from generator.model import DecoderOnlyTacticGenerator, DummyTacticGenerator


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
                        is_root=True
                    )
                    self.nodes = {init_state: self.root}
                    self.priority_queue = [self.root]
                    self.shared_node = 0
                    self.incorrect_count = []
                    self.incorrect_node_count = []

                    with torch.no_grad():
                        try:
                            with logger.contextualize(root_level=0):
                                self.begin_time = time.monotonic()
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
                
                sorry_flag = False
                if self.root.status == Status.PROVED:
                    proof = self.extract_proof2(self.root)
                else:
                    proof = None
                
                if proof:
                    flag = False
                    if len(proof) > 390:
                        a_proof, b_proof = proof[:50], proof[50:]
                        random.shuffle(b_proof)
                        proof = a_proof + b_proof
                    for p in proof[:390]:
                        if dojo.check_proof(p):
                            proof = p
                            flag = True
                            break
                    if not flag:
                        self.root.status = Status.FAKE_PROVED
                        proof = proof[0]
                
                if sorry_flag:
                    proof.append("sorry!")


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
        sub_search_time = 0

        while True:
            if len(priority_queue) == 0 or root_level > 10:
                logger.info(f"Ran out of nodes to search or root level {root_level} too deep")
                root.status = Status.FAILED
                in_nodes_to_update = root.boardcast_failure()
                for in_node in in_nodes_to_update*2:
                    in_node._recompute_status()
                for edge in root.in_edges:
                    edge.src._recompute_status()
                break

            if root_level > 0 and time.monotonic() - time_start - sub_search_time >= 120:
                logger.info(f"Current level {root_level} is searching for too long: {time.monotonic() - time_start - sub_search_time}, subsearch_time: {sub_search_time}, returning back")
                root.status = Status.FAILED
                in_nodes_to_update = root.boardcast_failure()
                for in_node in in_nodes_to_update*2:
                    in_node._recompute_status()
                for edge in root.in_edges:
                    edge.src._recompute_status()
                break

            try:
                # TODO: I not sure this is right thing to do, origianl one is in [Status.OPEN, Status.HalfProved]
                if root.status == Status.OPEN:
                    self._step(root, nodes, priority_queue)
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - self.begin_time
            print("time:", self.total_time)
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
                c_root, c_traj, succes_trajs = self._get_next_sorry_root(root)
                logger.debug(f"Choosen root: {c_root}")
                logger.debug(f"Choosen trajectory: {c_traj}")
                logger.debug(f"There are {len(succes_trajs)} with choosen root, here listed the first 10")
                for traj in succes_trajs[:10]:
                    logger.debug(traj)
                c_nodes = {c_root.state: c_root}
                c_priority_queue = [c_root]
                logger.info(f">>> Root {root} found a proof, going deeper with root {c_root}")
                with logger.contextualize(root_level=root_level+1):
                    sub_search_start_time = time.monotonic()
                    self._best_first_search(c_root, c_nodes, c_priority_queue, root_level=root_level+1)
                    sub_search_time += time.monotonic() - sub_search_start_time

    def _get_next_sorry_root(self, root: InternalNode):
        def __filter_failed_trajectory(trajectory):
            for obj in trajectory.traj:
                if isinstance(obj, Node):
                    assert obj.status in [Status.PROVED, Status.HALF_PROVED]
                if isinstance(obj, SorryEdge):
                    if any([sorry_root.status == Status.FAILED for sorry_root in obj.sorry_roots]):
                        return False
            return True

        # collect success nodes
        success_nodes = self._collect_success_nodes(root)

        all_success_trajectories = []
        for success_node in success_nodes:
            success_trajectories = get_trajectory(success_node, stop_node=root, status_requirements=[Status.PROVED, Status.HALF_PROVED])
            success_trajectories = list(filter(__filter_failed_trajectory, success_trajectories))
            all_success_trajectories.extend(success_trajectories)
        assert len(all_success_trajectories) > 0, "No success trajectory"
        
        all_trajectory_sorry_roots = []
        for traj in all_success_trajectories:
            traj_sorry_roots = []
            for obj in traj.traj:
                if isinstance(obj, Node):
                    assert obj.status in [Status.PROVED, Status.HALF_PROVED]
                if isinstance(obj, SorryEdge):
                    assert not any([sorry_root.status == Status.FAILED for sorry_root in obj.sorry_roots]), \
                        f"Success trajectory contains failed sorry root: {traj}"
                    # if any([sorry_root.status == Status.FAILED for sorry_root in obj.sorry_roots]):
                    #     print("here")
                    traj_sorry_roots.extend(obj.sorry_roots)
            all_trajectory_sorry_roots.append(traj_sorry_roots)
        
        traj_and_sorry_roots = list(zip(all_success_trajectories, all_trajectory_sorry_roots))
        traj_and_sorry_roots = sorted(traj_and_sorry_roots, 
                                      key=lambda x: len([r for r in x[1] 
                                      if r.status in [Status.OPEN, Status.HALF_PROVED]]))
        
        choosen_traj, choosen_sorry_roots = traj_and_sorry_roots[0]
        
        # last first stragies
        choosen_sorry_root = [r for r in choosen_sorry_roots if r.status in [Status.OPEN, Status.HALF_PROVED]][-1]

        trajectory_with_choosen_root = []
        for trajectory, sorry_roots in traj_and_sorry_roots:
            if choosen_sorry_root in sorry_roots:
                trajectory.marked_sorry_root = choosen_sorry_root
                trajectory_with_choosen_root.append(trajectory)
        
        return choosen_sorry_root, choosen_traj, trajectory_with_choosen_root

    def extract_proof2(self, root):
        def __filter_failed_trajectory(trajectory):
            for obj in trajectory.traj:
                if isinstance(obj, Node):
                    assert obj.status == Status.PROVED
                if isinstance(obj, SorryEdge):
                    if any([sorry_root.status != Status.PROVED for sorry_root in obj.sorry_roots]):
                        return False
            return True

        # collect success nodes
        success_nodes = self._collect_success_nodes(root)

        all_success_trajectories = []
        for success_node in success_nodes:
            success_trajectories = get_trajectory(success_node, stop_node=root, status_requirements=[Status.PROVED])
            success_trajectories = list(filter(__filter_failed_trajectory, success_trajectories))
            all_success_trajectories.extend(success_trajectories)
        assert len(all_success_trajectories) > 0, "No proven trajectory"

        final_trajs = []
        for traj in all_success_trajectories:
            if not any([isinstance(t, SorryEdge) for t in traj.traj]):
                str_traj = [t.tactic.strip() for t in traj.traj if isinstance(t, Edge)]
                final_trajs.append(str_traj)
                continue

            current_traj = [[]]
            traj_edges = [t for t in traj.traj if isinstance(t, Edge)]
            for edge in traj_edges:
                if isinstance(edge, SorryEdge):
                    all_tac = edge.tactic.split("sorry")
                    for idx, sorry_root in enumerate(edge.sorry_roots):
                        assert sorry_root.status == Status.PROVED
                        root_proof = self.extract_proof2(sorry_root)
                        for ct in current_traj:
                            ct.append(all_tac[idx].strip())
                        new_current_traj = []
                        for rp in root_proof:
                            for ct in current_traj:
                                new_current_traj.append(ct + rp)
                        current_traj = new_current_traj
                    if len(all_tac[-1]) > 0:
                        for ct in current_traj:
                            ct.append(all_tac[-1])
                else:
                    for ct in current_traj:
                        ct.append(edge.tactic)
            final_trajs.extend(current_traj)
        
        return final_trajs

    
    def _collect_success_nodes(self, root):
        success_node = []
        node_queue = Queue()
        node_queue.put(root)
        while not node_queue.empty():
            node = node_queue.get()
            if isinstance(node, ProofFinishedNode) and node.status == Status.PROVED and node not in success_node:
                success_node.append(node)
            if not isinstance(node, InternalNode):
                continue
            for edge in node.out_edges:
                if edge.dst.status in [Status.PROVED, Status.HALF_PROVED]:
                    node_queue.put(edge.dst)
        return success_node

    def _get_single_history(self, node):
        history = []
        current_node = node
        while len(current_node.in_edges) > 0:
            edge = current_node.in_edges[0]
            if isinstance(edge, SorryEdge):
                if edge.dst == current_node:
                    history.append(edge.tactic)
                else:
                    assert current_node in edge.sorry_roots
                    sorry_tactic = None
                    for i, r in enumerate(edge.sorry_roots):
                        if current_node == r:
                            sorry_tactic = edge.sorry_tactics[i]
                    assert sorry_tactic is not None
                    history.append(sorry_tactic)
            else:
                history.append(edge.tactic)
            current_node = edge.src
        history.append(self.root.state.from_tactic)
        history =  list(reversed(history))
        expanded_history = [history[0]]
        for his in history[1:]:
            his = his.split("\n")
            expanded_history.extend(his)
        return expanded_history

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
        logger.debug(f"Privous Step: {get_trajectory(search_node)[0]}")
        if search_node.status == Status.FAILED:
            logger.debug("Node is already failed, returning...")
            return


        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
            from_tactic = "\n".join(self._get_single_history(search_node)[-self.history_size:])
        else:
            # ts = search_node.state.unsolved_tactic_state
            assert False, "Why this will happen?"
        suggestions = self._generate_tactics(ts, context=from_tactic)

        # if self.check_share_tactics(search_node, suggestions) > 0:
        #     self.incorrect_node_count.append(search_node)
        # # self.incorrect_count += incorrect_count
        # logger.warning(f"Incorrect shared node count: {len(self.incorrect_count)}")

        if '''show "\\<And>n. deg R f < n \\<Longrightarrow> n_mult f n = \\<zero>" sorry''' in [t[0] for t in suggestions]:
            print("herere")

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
    
    def check_share_tactics(self, node: InternalNode, tactics):
        if len(node.in_edges) < 2:
            return 0
        
        all_result_states = []
        for edge in node.in_edges:
            cur_state = self.dojo.run_tac(edge.src.state, tactic=edge.tactic)
            assert cur_state == node.state
            # assert actual_tac == edge.tactic

            result_states = [self.dojo.run_tac(cur_state, tactic=tac[0]) for tac in tactics]
            all_result_states.append(result_states)
        for gold_result in all_result_states:
            other_result = all_result_states
            incorrect_count = 0
            for results in other_result:
                for idx, state in enumerate(results):
                    if state != gold_result[idx]:
                        logger.warning(f"State not equal to gold result, {state}, gold: {gold_result[idx]}")
                        self.incorrect_count.append([state, gold_result[idx]])
                        incorrect_count += 1
        return incorrect_count
            

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
        sorry_responses = []
        sorry_tactics = []
        if (isinstance(response, TacticState) or isinstance(response, ProofFinished)) and "sorry" in tactic:
            tactics = tactic.split("sorry")
            # tactics = tactics if len(tactics[-1]) > 0 else tactics[:-1]
            for i in range(len(tactics)-1):
                if len(tactics[i]) <= 0:
                    logger.warning(f"No sorry tactic contains empty string: {tactics}")
                    return None
                sorry_tactic = "sorry".join(tactics[:i+1]).strip()
                sorry_response = self.dojo.run_tac(node.state, sorry_tactic, root.state)
                # if actual_sorry_tac != sorry_tactic:
                #     sorry_response = IsabelleError(f"actual_sorry_tac != sorry_tactic: {actual_sorry_tac}, {sorry_tactic}")
                
                # if the sorry tactic failed, we don't need the original tactic either.
                if isinstance(sorry_response, IsabelleError):
                    response = sorry_response
                    sorry_tactics, sorry_responses = [], []
                    break

                sorry_tactics.append(sorry_tactic)
                sorry_responses.append(sorry_response)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed
        trajctory_to_node = get_trajectory(node)

        result_nodes = []
        add_queue_flag = False
        for idx, resp in enumerate([response] + sorry_responses):
            if idx > 0:
                assert isinstance(resp, TacticState), f"sorry root is not TacticState: {resp}"

            try:
                # If we've seen this response before, use the existing node
                result_node = nodes[resp]

                # check to aviod loop
                if any([result_node in traj for traj in trajctory_to_node]):
                    return None

                if isinstance(resp, ProofFinished):
                    log_result(f"Result: proof successed! - {str(resp.message)}")
                elif isinstance(resp, IsabelleError):
                    log_result(f"Result: tactic failed! - {str(resp)}")
                else:
                    log_result(f"Result: duplicate result ! - {str(resp.pp)}")
            except KeyError:
                # Build a new node
                if isinstance(resp, ProofFinished):
                    result_node = ProofFinishedNode(resp)
                    log_result(f'Result: proof successed! - {str(resp.message)}')
                    logger.info("hererererer!!!SSSS!!!!!!!!!!!!!!!")
                elif type(resp) in (
                    IsabelleError,
                    TimeoutError,
                    ProofGivenUp,
                ):
                    result_node = ErrorNode(resp)
                    log_result(f'Result: tactic failed! - {str(resp)}')
                else:
                    assert isinstance(resp, TacticState)
                    result_node = InternalNode(
                        state=resp,
                        cumulative_logprob=logprob + node.cumulative_logprob if idx == 0 else 0,
                        is_root=idx!=0
                    )
                    log_result(f'Result: tactic success! - {resp.pp}')
                
                if idx == 0 and result_node.status == Status.OPEN:
                    add_queue_flag = True

            
            result_nodes.append(result_node)

        result_node, sorry_roots = result_nodes[0], result_nodes[1:]
        
        # if the tactic gives a root node, don's search here, return None
        if isinstance(result_node, InternalNode) and result_node.is_root:
            return None
        
        # if root node later bump into a non-root node, we also don't make it a root node.
        if any(sr.is_root == False for sr in sorry_roots):
            return None

        # Record the new node and add it to the search queue.
        for idx, resp in enumerate([response] + sorry_responses):
            nodes[resp] = result_nodes[idx]

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.  
        edge = Edge(tactic=tactic, src=node, dst=result_node)
        if len(sorry_roots) > 0:
            edge = SorryEdge(
                tactic=tactic,
                src=node,
                dst=result_node,
                sorry_tactics=sorry_tactics,
                sorry_roots=sorry_roots,
            )

        for res_node in result_nodes:
            res_node.in_edges.append(edge)
            self.shared_node += 1
            # logger.debug(f"Number of shared node: {self.shared_node}")
        
        if add_queue_flag:  # Don't search proved/failed nodes
            heapq.heappush(priority_queue, result_node)  # type: ignore

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
        use_sampling: bool,
        history_size: int,
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
                model_name_or_path=ckpt_path, use_sampling=use_sampling, device=torch.device("cpu")
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
                tac_gen = DummyTacticGenerator(
                    data_path="/hpc2hdd/home/zyang398/wanghaiming/data/isabelle/multilevel/test.json"
                )
            else:
                device = torch.device("cuda") if with_gpus else torch.device("cpu")
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


def check_tree(node):
    nodelist = Queue()
    nodelist.put(node)
    while not nodelist.empty():
        cur_node = nodelist.get()
        if isinstance(cur_node, InternalNode) and cur_node.status == Status.FAILED and cur_node.out_edges is not None:
            for dedge in cur_node.out_edges:
                if dedge.dst.status != Status.FAILED:
                    assert isinstance(dedge, SorryEdge)
                    assert any([rootd.status == Status.FAILED for rootd in dedge.sorry_roots])
        if isinstance(cur_node, InternalNode) and cur_node.out_edges is not None:
            for dedge in cur_node.out_edges:
                nodelist.put(dedge.dst)
