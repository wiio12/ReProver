"""Lightning module for the tactic generator."""
from collections import defaultdict
import json
import random
import torch
import openai
from loguru import logger
from torchmetrics import Metric
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from transformers import  AutoTokenizer, AutoModelForCausalLM, GPTNeoXPreTrainedModel, GPTNeoXTokenizerFast

def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"

def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")




torch.set_float32_matmul_precision("medium")


class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos ,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class DecoderOnlyTacticGenerator(TacticGenerator):
    def __init__(
        self,
        model_name_or_path,
        max_inp_seq_len: int = 840,
        max_oup_seq_len: int = 256,
        length_penalty: float = 0.0,
        use_sampling: bool = False,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.length_penalty = length_penalty
        self.use_sampling = use_sampling
        self.temperature = temperature
        self.device = device

        # we haven't train retrieval augmented generator yet
        self.retriever = None
        self.generator = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        if isinstance(self.generator, GPTNeoXPreTrainedModel):
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def generate(
        self,
        state: str,
        context: str = "",
        num_samples: int = 8,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            [state], [context], num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        context: List[str],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        if self.retriever is not None:
            raise NotImplementedError

        if all([len(c)==0 for c in context]):
            prompt = [f"GOAL {s} STEP\n" for s in state]
        else:
            prompt = [f"CONTEXT {ctx} GOAL {s} STEP\n" for ctx, s in zip(context, state)]
        logger.debug(prompt)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        tokenized_state = self.tokenizer(
            prompt,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        raw_scores = None
        # Generate tactic candidates using beam search.
        if self.use_sampling is False:
            output = self.generator.generate(
                input_ids=state_ids,
                attention_mask=state_mask,
                max_length=self.max_inp_seq_len + self.max_oup_seq_len,
                num_beams=num_samples,
                length_penalty=self.length_penalty,
                do_sample=False,
                num_return_sequences=num_samples,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            output = self.generator.generate(
                input_ids=state_ids,
                attention_mask=state_mask,
                max_length=self.max_inp_seq_len + self.max_oup_seq_len,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=num_samples,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_sequences = output.sequences[:, state_ids.size(1):]
            # logits to probs
            probs = torch.stack(output.scores, dim=1).softmax(-1)
            # now we need to collect the probability of the generated token
            # we need to add a dummy dim in the end to make gather work
            gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
            # calculate non eos token mask, we exclude the token probabilities for eos tokens.
            non_eos_token_mask = gen_sequences != self.tokenizer.eos_token_id
            log_probs = (gen_probs.log().nan_to_num(0) * non_eos_token_mask).sum(-1)
            # Add small random Gaussian noise to prevent priority queue errors
            log_probs += 0.00001 * torch.randn_like(log_probs)
            raw_scores = log_probs.tolist()

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        # extract the tactic
        tactic_outputs = []
        for rot in raw_output_text:
            if "STEP\n" in rot:
                tactic_outputs.append(rot[rot.index("STEP\n") + len("STEP\n"):])
            else:
                # the input is too long and the proofstep is trucated, thus no meaningful tactic will generate. we use some easy dummy to replace.
                tactic_outputs.append("by auto")
        raw_output_text = tactic_outputs

        raw_scores = output.sequences_scores.tolist() if raw_scores is None else raw_scores
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(sorted(
                list(zip_strict(output_text, output_score)),
                key=lambda x:x[1],
                reverse=True)
            )

        return tactics_with_scores


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in Lean3 theorem proofs. We are trying to solve the Lean3 theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(
                tactics_with_scores, key=lambda x: x[1], reverse=True
            )[: min(num_samples, len(tactics_with_scores))]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(
                f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}"
            )
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip_strict(
                state, file_path, theorem_full_name, theorem_pos
            )
        ]


class FixedTacticGenerator(TacticGenerator):
    def __init__(self, tactic, module) -> None:
        self.tactic = tactic
        self.module = module

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return [(f"{{ {self.tactic} }}", 1.0)]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos,
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, tfn, tp, num_samples)
            for s, f, tfn, tp in zip(state, file_path, theorem_full_name, theorem_pos)
        ]



class DummyTacticGenerator(TacticGenerator):
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path
        with open(data_path, "r") as f:
            data = json.load(f)
        self.io_pair = defaultdict(list)
        for line in data:
            self.io_pair[line[0]].append(line[1])
    
    def generate(
        self,
        state: str,
        context: str = "",
        num_samples: int = 8,
    ) -> List[Tuple[str, float]]:
        prompt = f"CONTEXT {context} GOAL {state} STEP"
        answer = []
        if prompt in self.io_pair:
            answer = self.io_pair[prompt]
        answer_with_score = []
        for a in answer:
            answer_with_score.append((a, -random.random()))
        return answer_with_score

    def batch_generate(
        self,
        state: List[str],
        context: List[str],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError