"""Definitions of the search tree used by the prover.
"""
import math
from enum import Enum

import numpy as np
from multilevel_isabelle.src.main.python.pisa_client import (
    TacticState,
    IsabelleError,
    TimeoutError,
    ProofGivenUp,
    ProofFinished,
)
from abc import ABC, abstractmethod
from functools import total_ordering
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterable, Union
from loguru import logger


class Status(Enum):
    """Status of a node or a proof search."""
    HALF_PROVED = "HalfProved"   # This node is half proved, with complete verifiable sketeches.
    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.
    FAKE_PROVED = "FakeProved"

class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError


@dataclass
class ProofFinishedNode(Node):
    inner: ProofFinished
    status = Status.PROVED
    distance_to_proof = 0
    is_terminal = True
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )
    def boardcast_failure(self):
        self.status = Status.FAILED
        return [edge.src for edge in self.in_edges]

@dataclass
class ErrorNode(Node):
    inner: Union[IsabelleError, TimeoutError, ProofGivenUp]
    status = Status.FAILED
    distance_to_proof = math.inf
    is_terminal = True
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )
    def boardcast_failure(self):
        return [edge.src for edge in self.in_edges]


@total_ordering
@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_ priority than node_b.
    """

    # Goal state this node represents. Two nodes are considered equal if their states
    # are equal; this is the only hashed field and must not be changed.
    state: TacticState = field(compare=True)

    # The sum of action logprobs along edges from the root to this node
    cumulative_logprob: float = field(compare=False, repr=False)

    is_root: bool = field(default=False, compare=False, repr=False)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and must not change after that.
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )

    # A node is proved if any child is proved, and failed if every child is failed
    # (or there are no children). A node that is proved or failed cannot change status
    # because nothing is ever added to out_edges. _status is recomputed on an as-needed
    # basis by children, since proving or failing a child may prove or fail this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False  # type: ignore[override]

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    @property
    def out_edges(self):
        return self._out_edges

    # This setter implements exploring this node
    @out_edges.setter
    def out_edges(self, out_edges: Iterable["Edge"]) -> Optional[List["Edge"]]:
        if self.is_explored:
            raise RuntimeError("Node is already explored.")

        self._out_edges = list(out_edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children. Explored nodes are never re-searched.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s):
        self._status = s

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.is_explored and self.out_edges is not None

        # # If this node is proved or failed, nothing can change that
        # if self._status not in [Status.OPEN, Status.HALF_PROVED]:
        #     return
        if self._status == Status.FAILED:
            for edge in self.out_edges:
                if edge.dst.status != Status.FAILED:
                    assert isinstance(edge, SorryEdge)
                    assert any([root.status == Status.FAILED for root in edge.sorry_roots])
            return
        
        # merge status in sorry edge
        # temp_dst_nodes = []
        merged_status = []
        re_opened = False
        for idx, edge in enumerate(self.out_edges):
            if isinstance(edge, SorryEdge):
                # if sub-root is proved, it have no effect on the status of dst node
                if all([root.status == Status.PROVED for root in edge.sorry_roots]):
                    # edge.dst.status = edge.dst.status
                    merged_status.append(edge.dst.status)
                # if sub-root is failed, it fails the dst node. Only true when sub-root is not allow to revisit!
                elif any([root.status == Status.FAILED for root in edge.sorry_roots]) and edge.dst.status != Status.FAILED:
                    # new_failed_nodes_ori_status.append(edge.dst.status)
                    # edge.dst.status = Status.FAILED
                    # new_failed_nodes.append(edge.dst)
                    merged_status.append(Status.FAILED)
                # if sub-root is open or half-proved, it does not affect the status of dst node, but creates a special half-flag.
                elif any([root.status in [Status.OPEN, Status.HALF_PROVED] for root in edge.sorry_roots]):
                    # for implementation convenience, we use half-proved to represent the sorry edge
                    if edge.dst.status == Status.PROVED:
                        # edge.dst.status = Status.HALF_PROVED
                        # temp_dst_nodes.append(edge.dst)
                        merged_status.append(Status.HALF_PROVED)
                    else:
                        merged_status.append(edge.dst.status)
                else:
                    merged_status.append(edge.dst.status)
            else:
                merged_status.append(edge.dst.status)
        
        assert len(merged_status) == len(self.out_edges)

        # If any child is half-proved (path to this node contains sorry edge), 
        # this node is half-proved, and so are parents recursively
        if any(status == Status.HALF_PROVED for status in merged_status):
            self._status = Status.HALF_PROVED
        else:
            if self._status != Status.OPEN:
                self._status = Status.OPEN
                re_opened = True

        # If any children are proved, and no, this node is proved. This may prove some parents too.
        if any(status == Status.PROVED for status in merged_status):
            self._status = Status.PROVED
            
        # If all children failed, this node is failed. This may fail some parents too.
        if all(status == Status.FAILED for status in merged_status):
            self._status = Status.FAILED

        # reset the temp dst nodes
        # for node in temp_dst_nodes:
        #     node.status = Status.PROVED

        # # restore the temp failure
        # for idx, node in enumerate(new_failed_nodes):
        #     node.status = new_failed_nodes_ori_status[idx]
            # in_nodes_to_update = node.boardcast_failure()
            # # print("in_nodes_to_update:", len(in_nodes_to_update))
            # for in_node in in_nodes_to_update*2:
            #     # if in_node != self:
            #     in_node._recompute_status()

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self._status != Status.OPEN or re_opened:
            for edge in self.in_edges:
                edge.src._recompute_status()

    def boardcast_failure(self):
        in_nodes_to_updates = []
        if self.status == Status.FAILED and self.out_edges is not None:
            for edge in self.out_edges:
                edge.dst.status = Status.FAILED
                in_nodes = edge.dst.boardcast_failure()
                in_nodes_to_updates.extend(in_nodes)

            # It seems like we have to update the parent as well. Chile: Proved -> Failure, Parent: Proved/HalfProved -> Open/Failure
        if self.status == Status.FAILED and self.in_edges is not None:
            for edge in self.in_edges:
                in_nodes_to_updates.append(edge.src)

        # TODO: How to deal with multiple sorry root? do we need to boardcast to other sorry node
        # we might not update the other sorry node, other sorry node may also be use in other sorry edge that only them exist.

        return in_nodes_to_updates

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self):
        """
        Recursively update the distance_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()

    # NOTE: Nodes are compared by _negative_ priority, to make heapq act as a max-priority-queue.
    @property
    def priority(self) -> float:
        return self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority


    def extract_proof(self):
        """
        Extract a proof of the current node as a sequence of edges.
        """
        if self.status != Status.PROVED:
            return None, False
        assert self.is_explored

        # proving_edge = min(
        #     self.out_edges,
        #     key=Edge.distance_to_proof,
        # )
        proving_edge = self.out_edges[
            np.argmin([edge.distance_to_proof() for edge in self.out_edges])
        ]

        proof = []
        sorry_flag = False
        if isinstance(proving_edge, SorryEdge):
            sorry_flag = True
            all_tac = proving_edge.tactic.split("sorry")
            for idx, sorry_root in enumerate(proving_edge.sorry_roots):
                root_proof, _ = sorry_root.extract_proof()
                proof.append(all_tac[idx])
                proof.extend(root_proof)
            if len(all_tac[-1]) > 0:
                proof.append(all_tac[-1])
        else:
            proof.append(proving_edge.tactic)
        

        if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert isinstance(proving_edge.dst, ProofFinishedNode)
            return proof, sorry_flag
        else:
            # Recursive case: prove the child, then add this edge
            assert isinstance(proving_edge.dst, InternalNode)
            child_proof, dst_sorry_flag = proving_edge.dst.extract_proof()
            sorry_flag = dst_sorry_flag or sorry_flag
            assert child_proof
            return proof + child_proof, sorry_flag


    #########
    # Debug #
    #########

    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        if not self.is_explored:
            assert self.status == Status.OPEN
            return  # Nothing more can be said about unexplored nodes

        for edge in self.in_edges:
            if isinstance(edge, SorryEdge):
                assert edge.dst is self or edge.sorry_root is self
            else:
                assert edge.dst is self

        if self.out_edges == []:
            assert self.status == Status.FAILED
        else:
            for edge in self.out_edges:  # type: ignore
                assert edge.src is self

        if self.status == Status.PROVED:
            assert self.out_edges
            assert any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert all(edge.dst.status in [Status.PROVED, Status.HALF_PROVED] for edge in self.in_edges)

            proof_by_steps = self.extract_proof()
            assert proof_by_steps is not None
            assert self.distance_to_proof == len(proof_by_steps)

        elif self.status == Status.FAILED:
            assert self.out_edges is not None
            assert all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() == None
        elif self.status == Status.OPEN:
            assert self.out_edges
            assert not any(edge.dst.status == Status.PROVED for edge in self.out_edges)
            assert not all(edge.dst.status == Status.FAILED for edge in self.out_edges)
            assert self.distance_to_proof == math.inf
            assert self.extract_proof() == None

@dataclass
class Edge:
    """An edge in the search tree, representing a tactic."""

    tactic: str
    src: InternalNode = field(repr=False)
    dst: Node = field(repr=False)

    def distance_to_proof(self) -> float:
        return 1 + self.dst.distance_to_proof

@dataclass
class SorryEdge(Edge):

    sorry_tactics: List[str]
    sorry_roots: List[InternalNode] = field(repr=False)

    def distance_to_proof(self) -> float:
        return sum([sr.distance_to_proof for sr in self.sorry_roots])+ 1 + self.dst.distance_to_proof

@dataclass
class Trajectory:
    traj: List[Union[Node, Edge]]
    marked_sorry_root : InternalNode = field(default=None, compare=False)

    def __repr__(self) -> str:
        traj = []
        for elem in self.traj:
            if isinstance(elem, SorryEdge):
                if self.marked_sorry_root in elem.sorry_roots:
                    tactics = elem.tactic.split("sorry")
                    all_tac = []
                    for i in range(len(tactics)-1):
                        all_tac.append(tactics[i])
                        if elem.sorry_roots[i] == self.marked_sorry_root:
                            all_tac.append("**sorry**")
                        else:
                            all_tac.append("sorry")
                    all_tac.append(tactics[-1])
                    tactics_with_marked_sorry = "".join(all_tac)
                    traj.append(tactics_with_marked_sorry)
                else:
                    traj.append(elem.tactic)
            elif isinstance(elem, Edge):
                traj.append(elem.tactic)
            else:
                pass
        return f"Trajectory({str(traj)})"

    def __add__(self, o: "Trajectory"):
        return Trajectory(self.traj + o.traj)

    def __radd__(self, o: "Trajectory"):
        return Trajectory(o.traj + self.traj)
    
    def __contains__(self, key):
        return key in self.traj

    def reverse(self):
        self.traj.reverse()

def get_trajectory(begin_node, stop_node=None, status_requirements=None, recursive_level=0):
        if recursive_level > 500:
            logger.warning("The recursive loop is larger than 500")

        if len(begin_node.in_edges) == 0 or begin_node == stop_node:
            return [Trajectory([begin_node])]
        
        all_trajectories = []
        for edge in begin_node.in_edges:
            if status_requirements and edge.src.status not in status_requirements:
                continue
            trajs_to_node = get_trajectory(
                edge.src, 
                status_requirements=status_requirements, 
                stop_node=stop_node, 
                recursive_level=recursive_level+1
            )
            for traj in trajs_to_node:
                all_trajectories.append(traj + Trajectory([edge, begin_node]))
        return all_trajectories