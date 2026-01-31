import argparse
import concurrent
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List
import concurrent.futures
import tiktoken
import openai
from tenacity import *
from openai import OpenAI, RateLimitError, APIConnectionError, APIError, BadRequestError
from tqdm import tqdm
from dotenv import load_dotenv

from editbench.config import SRC_INF_BENCHMARK_DATA
from editbench.inference.constants import EXPERIMENTAL_RESULTS
from editbench.inference.utils import extract_diff
from editbench.inference.prompt_builder import remove_last_file_from_prompt
from editbench.utils.dataset_utils import get_inf_datasets

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_LIMITS = {
    "claude-instant-1": 100_000,
    "claude-2": 100_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "qwen3-235b-a22b": 128_000,
    "gemini-2.5-pro": 1_000_000,
    "deepseek-v3.2": 128_000,
    "deepseek-reasoner": 128_000,
    "gpt-3.5-turbo-16k-0613": 16_385,
    "gpt-3.5-turbo-0613": 4_097,
    "gpt-3.5-turbo-1106": 16_385,
    "gpt-4-32k-0613": 32_768,
    "gpt-4-0613": 8_192,
    "gpt-4-1106-preview": 128_000,
    "gpt-4-0125-preview": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 128_000,
    "o1-preview": 128_000,
    "deepseek-chat": 128_000,  # updated to DeepSeek-V3.2
    "gpt-5-codex": 400_000
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "claude-sonnet-4-5-20250929": 0.000003,  # $3 / 1M tokens
    "qwen3-235b-a22b": 0.000000556,  # $0.556 / 1M tokens (¥0.004 / 1k tokens converted at 7.2 CNY/USD)
    "gemini-2.5-pro": 0.00000125,  # $1.25 / 1M tokens (<=200k tokens), $2.50 / 1M tokens (>200k tokens)
    "deepseek-v3.2": 0.000000222,  # ¥1.6 / 1M tokens 
    "deepseek-reasoner": 0.000000222,  # ¥1.6 / 1M tokens 
    "deepseek-chat": 0.000000222,  # ¥1.6 / 1M tokens
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,  # probably still 0613
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    "gpt-4o": 0.0000025,
    "gpt-4.1": 0.000002,  # $2 / 1M tokens
    "o1-preview": 0.000015,
    "gpt-5-codex": 0.00000125
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "claude-sonnet-4-5-20250929": 0.000015,  # $15 / 1M tokens
    "qwen3-235b-a22b": 0.000005556,  # $5.556 / 1M tokens (¥0.04 / 1k tokens converted at 7.2 CNY/USD)
    "gemini-2.5-pro": 0.00001,  # $10.00 / 1M tokens (<=200k tokens), $15.00 / 1M tokens (>200k tokens)
    "deepseek-v3.2": 0.000000333,  # ¥2.4 / 1M tokens 
    "deepseek-reasoner": 0.000000333,  # ¥2.4 / 1M tokens 
    "deepseek-chat": 0.000000333,  # ¥2.4 / 1M tokens 
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    "gpt-4o": 0.00001,
    "gpt-4.1": 0.000008,  # $8 / 1M tokens
    "o1-preview": 0.00006,
    "gpt-5-codex": 0.00001
}

# used for openai
ENGINES = {
    "gpt-3.5-turbo-16k-0613": "gpt-35-turbo-16k",
    "gpt-4-0613": "gpt-4",
    "gpt-4-32k-0613": "gpt-4-32k",
    "gpt-4o": "gpt-4o",
    "gpt-5-codex": "gpt-5-codex"
}

# mapping of model to encoding
# Note: tiktoken mainly supports the encoding of OpenAI models (cl100k_base is used for GPT-4/3.5)
# For other models (Claude, Qwen, Gemini, DeepSeek), their tokenizer is different from OpenAI
# here we use cl100k_base as an approximation, which may not be accurate
# if you need more accurate counting, it is recommended to use the tokenizer of each model (such as anthropic SDK, qwen tokenizer, etc.)
MODEL_ENCODING = {
    # GPT series: use tiktoken to automatically identify (most accurate)
    "gpt-5-codex": None,  # None means use tiktoken.encoding_for_model() to automatically identify
    "gpt-4.1": None,  # None means use tiktoken.encoding_for_model() to automatically identify
    
    # third-party models: use cl100k_base as an approximation (not accurate, but can be used as a reference)
    "deepseek-v3.2": "cl100k_base",  # Note: DeepSeek has its own tokenizer, this is just an approximation
    "qwen3-235b-a22b": "cl100k_base",  # Note: Qwen has its own tokenizer, this is just an approximation
    "gemini-2.5-pro": "cl100k_base",  # Note: Gemini uses Google's tokenizer, this is just an approximation
    "claude-sonnet-4-5-20250929": "cl100k_base",  # Note: Claude uses Anthropic's tokenizer, this is just an approximation
}

MAP_MODEL_TO_COFIG = {
    "gpt-5-codex": {
        "temperature": 1,
        "top_p": 0.95,
        "n": 1,
    },
    "swe-ce-agent/gpt-5-codex": {
        "temperature": 1,
        "top_p": 0.95,
        "n": 1,
    },
    "agentless-ce/gpt-5-codex": {
        "temperature": 1,
        "top_p": 0.95,
        "n": 1,
    },
    "cursor": {
        "temperature": 1,
        "n": 1,
    },
}

MAP_MODEL_TO_COFIG.update({
    key: {
        "temperature": 0,
        "top_p": 0.95,
        "n": 1,
    }
    for key in ["qwen3-235b-a22b", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "deepseek-v3.2", "gpt-4.1", "swe-ce-agent/qwen3-235b-a22b", 
    "swe-ce-agent/claude-sonnet-4-5-20250929", "swe-ce-agent/gemini-2.5-pro", "swe-ce-agent/deepseek-v3.2", "swe-ce-agent/gpt-4.1", 
    "agentless-ce/qwen3-235b-a22b", "agentless-ce/claude-sonnet-4-5-20250929", "agentless-ce/gemini-2.5-pro", "agentless-ce/deepseek-v3.2", "agentless-ce/gpt-4.1"]
}
)


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def claude_tokenize(string: str, api) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = api.count_tokens(string)
    return num_tokens


def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    cost = (
            MODEL_COST_PER_INPUT[model_name] * input_tokens
            + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost



# because BadRequestError needs the response object, we create a simple exception class
class CustomBadRequestError(BadRequestError):
    def __init__(self, message, code):
        # do not call the parent class __init__, directly set the attributes
        Exception.__init__(self, message)
        self.message = message
        self.code = code
        self.body = {'error': {'code': code, 'message': message}}
        self.response = None
    
    def __str__(self):
        return self.message

class ResponseWrapper:
    """wrap the OpenAI Chat Completions API response into a format compatible with responses.create()"""
    def __init__(self, completion):
        # check if completion.choices is None or empty
        if completion.choices is None or len(completion.choices) == 0:
            # try to get error information from model_extra
            error_message = "Unknown error"
            if hasattr(completion, 'model_extra') and completion.model_extra:
                error_info = completion.model_extra.get('error', {})
                error_message = error_info.get('message', 'Unknown error')
            
            # raise BadRequestError exception, code is 20059
            raise CustomBadRequestError(message=error_message, code='20059')
        
        self.output_text = completion.choices[0].message.content
        self.usage = completion.usage


def _prepare_chat_completions_kwargs(model_name: str, kwargs: dict, timeout: Optional[float] = None) -> dict:
    """
    prepare parameters for Chat Completions API
    
    :param model_name: model name
    :param kwargs: original parameters
    :param timeout: timeout
    :return: prepared parameters dictionary
    """
    # basic excluded parameters
    excluded_keys = ["n", "timeout"]
    
    # Claude model does not support top_p, only use temperature
    if "claude" in model_name.lower():
        excluded_keys.append("top_p")
        model_kwargs = {key: value for key, value in kwargs.items() if key not in excluded_keys}
        if "temperature" not in model_kwargs:
            model_kwargs["temperature"] = 0
    else:
        # Qwen, Gemini, DeepSeek support temperature and top_p
        model_kwargs = {key: value for key, value in kwargs.items() if key not in excluded_keys}
        if "temperature" not in model_kwargs:
            model_kwargs["temperature"] = 0
        if "top_p" not in model_kwargs:
            model_kwargs["top_p"] = 0.95
    
    # add timeout
    if timeout is not None:
        model_kwargs["timeout"] = timeout
    
    return model_kwargs


def _call_chat_completions_api(client, model_name: str, system_prompt: str, user_prompt: str, 
                                timeout: Optional[float] = None, **kwargs):
    """
    unify calling OpenAI Chat Completions API format models
    
    :param client: OpenAI client
    :param model_name: model name
    :param system_prompt: system prompt
    :param user_prompt: user prompt
    :param timeout: timeout
    :param kwargs: other parameters
    :return: wrapped response object
    """
    model_kwargs = _prepare_chat_completions_kwargs(model_name, kwargs, timeout)
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **model_kwargs
    )

    return ResponseWrapper(completion)


# @retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3))
def chat_openai(client, model_name, prompt, timeout: Optional[float] = None, **kwargs):
    """
    call OpenAI API for chat
    
    :param client: OpenAI client
    :param model_name: model name
    :param prompt: prompt (contains system and user messages)
    :param timeout: maximum response time (seconds), if None, use client default value
    :param kwargs: other model parameters (temperature, top_p, n, etc.)
    :return: (completion, cost) tuple
    """
    system_prompt = prompt.split("\n", 1)[0]
    user_prompt = prompt.split("\n", 1)[1]
    
    # extract timeout parameter, if exists, remove from kwargs
    if timeout is None:
        timeout = kwargs.pop("timeout", None)
    
    try:
        # gpt-5-codex uses responses.create() API
        if model_name == "gpt-5-codex":
            api_kwargs = {"timeout": timeout} if timeout is not None else {}
            completion = client.responses.create(
                model=model_name,
                instructions=system_prompt,
                input=user_prompt,
                store=False,
                **api_kwargs,
                **{key: value for key, value in kwargs.items() if key not in ["temperature", "top_p", "n", "timeout"]}
            )
        # Claude, Qwen, Gemini, DeepSeek uses standard Chat Completions API
        elif any(keyword in model_name.lower() for keyword in ["claude", "qwen", "gemini", "deepseek", "gpt-4.1"]):
            completion = _call_chat_completions_api(
                client, model_name, system_prompt, user_prompt, timeout, **kwargs
            )
        # other models use responses.create() API (default)
        else:
            # ensure temperature and top_p are from kwargs, if not, use default value
            temperature = kwargs.get("temperature", 0)
            top_p = kwargs.get("top_p", 0.95)
            
            # debug log: print the actual used parameters
            logger.debug(f"call responses.create() - model: {model_name}, temperature: {temperature}, top_p: {top_p}")
            logger.debug(f"kwargs: {kwargs}")
            
            api_kwargs = {"timeout": timeout} if timeout is not None else {}
            completion = client.responses.create(
                model=model_name,
                instructions=system_prompt,
                input=user_prompt,
                temperature=temperature,
                top_p=top_p,
                store=False,
                **api_kwargs,
                **{key: value for key, value in kwargs.items() if key not in ["temperature", "top_p", "n", "timeout"]}
            )
        if hasattr(completion, 'error') and completion.error is not None and hasattr(completion.error, 'code') and completion.error.code == "context_length_exceeded":
            raise CustomBadRequestError(message=completion.error.code, code='20059')
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = calc_cost(model_name, input_tokens, output_tokens)
        return completion, cost
    except openai.APIConnectionError as e:
        # traceback.print_exc()
        print("-----APIConnectionError, retrying-----")
        raise e
    except openai.RateLimitError as e:
        # traceback.print_exc()
        print("-----RateLimitError-----")
        raise e
    except openai.BadRequestError as e:
        # BadRequestError should be placed before Timeout, because if BadRequestError is a subclass of Timeout,
        # it will be captured by the except block of Timeout
        error_code = getattr(e, 'code', 'N/A')
        error_message = getattr(e, 'message', str(e))
        logger.error(f"BadRequestError in chat_openai for {model_name}: code={error_code}, message={error_message}")
        raise e
    except openai.APIError as e:
        # capture other APIError subclasses (such as InternalServerError, PermissionDeniedError, etc.)
        # these exceptions should be placed after Timeout, but before Exception
        error_type = type(e).__name__
        error_message = str(e)
        logger.error(f"APIError ({error_type}) in chat_openai for {model_name}: {error_message}")
        print(f"-----APIError ({error_type}): {error_message}-----")
        print(f"Exception type: {error_type}, MRO: {type(e).__mro__}")
        traceback.print_exc()
        raise e
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        logger.error(f"Unexpected Exception ({error_type}) in chat_openai for {model_name}: {error_message}")
        print(f"-----Unexpected Exception ({error_type}): {error_message}-----")
        print(f"Exception type: {error_type}, MRO: {type(e).__mro__}")
        traceback.print_exc()
        raise e


class InferenceWorker:
    """Inference worker thread class, each thread uses one API Key"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or os.getenv("BASE_URL")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.key_masked = api_key[:5] + '*' * max(5, len(api_key) - 10) + api_key[-5:]

    def run_inference(self, datum, model_name: str, model_args: Dict,
                      result_lock: threading.Lock, out_file: str, timeout: Optional[float] = None) -> Optional[float]:
        """
        execute inference for a single instance
        
        Args:
            datum: data instance
            model_name: model name
            model_args: model parameters
            result_lock: result write lock
            out_file: output file path
            timeout: maximum response time (seconds)
        """
        instance_id = datum.instance_id
        output_dict = {
            "instance_id": instance_id,
            "model_name": model_name,
            **model_args,
            "prompt": datum.prompt,
        }

        max_retries = 1
        retry_delay = 5
        current_prompt = datum.prompt  # save current prompt, in case it needs to be modified
        removed_file_retry = False  # mark if it is because of removing file and retry

        # use while loop to support retry after removing file
        attempt = 0
        while attempt < max_retries or removed_file_retry:
            # if it is because of removing file and retry, reset flag (only allow one extra retry)
            if removed_file_retry:
                removed_file_retry = False
            else:
                attempt += 1
            
            try:
                # record start time
                start_time = time.time()
                
                # call OpenAI API
                response, cost = chat_openai(
                    self.client,
                    model_name,
                    current_prompt,
                    timeout=timeout,
                    **model_args
                )
                
                # record end time and calculate response time
                end_time = time.time()
                response_time = end_time - start_time

                completion = response.output_text
                output_dict["full_output"] = completion
                output_dict["model_patch"] = extract_diff(completion)
                output_dict["cost"] = cost
                output_dict["response_time"] = response_time  # response time of the model (seconds)
                output_dict["prompt"] = current_prompt  # use the actual used prompt

                # thread-safe write result
                with result_lock:
                    with open(out_file, "a+") as f:
                        print(json.dumps(output_dict), file=f, flush=True)

                return cost

            except RateLimitError:
                print(f"⚠️  Key {self.key_masked} triggers rate limit, attempt {attempt}/{max_retries}...")
                time.sleep(retry_delay * attempt)  # exponential backoff
            except APIConnectionError:
                print(f"⚠️  Key {self.key_masked} connection error, attempt {attempt}/{max_retries}...")
                time.sleep(retry_delay * attempt)
            except BadRequestError as e:
                error_code = getattr(e, 'code', 'N/A')
                error_message = getattr(e, 'message', str(e))
                print(f"❌  Key {self.key_masked} BadRequestError (400): code={error_code}, message={error_message}")
                # logger.error(f"BadRequestError for instance {instance_id}: code={error_code}, message={error_message}")
                
                # if it is input length too long (20059), try to remove the last file
                if error_code == "20059":
                    new_prompt = remove_last_file_from_prompt(current_prompt)
                    if new_prompt == current_prompt:
                        # cannot delete file (already has only one file), write error and return
                        print(f"❌  Key {self.key_masked} cannot delete file, input context exceeds")
                        output_dict["status"] = "failed"
                        output_dict["error"] = "input context exceeded"
                        output_dict["prompt"] = current_prompt
                        with result_lock:
                            with open(out_file, "a+") as f:
                                print(json.dumps(output_dict), file=f, flush=True)
                        return None
                    else:
                        # successfully deleted file, update prompt and continue loop (retry)
                        print(f"⚠️  Key {self.key_masked} remove last file and retry")
                        logger.info(f"remove last file or compress file and retry instance {instance_id}")
                        current_prompt = new_prompt
                        output_dict["prompt"] = current_prompt
                        removed_file_retry = True  # mark need to retry (allow one extra loop)
                        continue  # continue loop, use new prompt to retry
                else:
                    # other BadRequestError, return directly, do not write to result file
                    return None
            # except openai.Timeout:
            #     print(f"⚠️  Key {self.key_masked} request timeout, attempt {attempt}/{max_retries}...")
            #     time.sleep(retry_delay * attempt)
            except APIError as e:
                print(f"❌  Key {self.key_masked} API error: {str(e)}")
                logger.error(f"APIError for instance {instance_id}: {str(e)}")
                # return directly, do not write to result file
                return None
            except Exception as e:
                print(f"❌  Key {self.key_masked} unexpected error when processing instance {instance_id}: {str(e)}")
                logger.error(f"Unexpected error for instance {instance_id}: {str(e)}")
                traceback.print_exc()
                # return directly, do not write to result file
                return None

        # all retries failed (only RateLimitError, APIConnectionError, Timeout will reach here)
        print(f"❌  Key {self.key_masked} processing instance {instance_id} failed after {max_retries} retries")
        output_dict["status"] = "failed"
        output_dict["error"] = f"Max retries ({max_retries}) exceeded"

        with result_lock:
            with open(out_file, "a+") as f:
                print(json.dumps(output_dict), file=f, flush=True)
        return 0


def inference_openai(inf_datasets, model_name, model_args, out_file, encoding=None,
                     existing_ids=None, max_workers: int = 5, openai_keys: Optional[List[str]] = None,
                     timeout: Optional[float] = None, target_instance_ids: Optional[List[str]] = None):
    """
    multi-threaded OpenAI inference, support multiple API Key load balancing

    Args:
        inf_datasets: inference dataset
        model_name: model name
        model_args: model parameters
        out_file: output file path
        encoding: token encoding (default auto-get)
        existing_ids: list of processed instance IDs (skip these)
        max_workers: maximum number of threads (default 5)
        openai_keys: list of OpenAI API Keys (default from environment variable)
        timeout: maximum response time (seconds), if None, use client default value
        target_instance_ids: list of target instance IDs (only process these IDs, if None, process all)
    """
    # initialize parameters
    if existing_ids is None:
        existing_ids = []
    if openai_keys is None:
        # get multiple keys from environment variable, separated by commas
        openai_keys = os.getenv("OPENAI_KEYS", os.getenv("OPENAI_KEY")).split(",")
        # remove duplicates and filter empty values
        openai_keys = [key.strip() for key in openai_keys if key.strip()]

    if not openai_keys:
        raise ValueError("no valid OpenAI API Key provided")

    # get and filter dataset
    # if target_instance_ids is specified, use it to filter
    if target_instance_ids is not None:
        inf_instances = get_inf_datasets(inf_datasets, instance_ids=target_instance_ids)
        print(f"📋 use target instance_ids to filter: {len(target_instance_ids)} target IDs")
    else:
        inf_instances = get_inf_datasets(inf_datasets)
    
    if encoding is None:
        # use encoding specified in MODEL_ENCODING dictionary first
        if model_name in MODEL_ENCODING:
            encoding_name = MODEL_ENCODING[model_name]
            if encoding_name is None:
                # None means use tiktoken to automatically recognize (for GPT series)
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                    logger.info(f"model {model_name} uses tiktoken to automatically recognize encoding: {encoding.name}")
                except KeyError:
                    # if automatic recognition fails, use cl100k_base as backup
                    encoding = tiktoken.get_encoding("cl100k_base")
                    logger.warning(f"model {model_name} cannot be automatically recognized by tiktoken, use cl100k_base as backup encoding")
            else:
                # use encoding specified in dictionary (usually cl100k_base as approximation)
                encoding = tiktoken.get_encoding(encoding_name)
                logger.warning(
                    f"model {model_name} uses specified encoding: {encoding_name} "
                    f"(note: for non-OpenAI models, this is just an approximation, may not be accurate)"
                )
        else:
            # if not specified in dictionary, try to use tiktoken to automatically recognize
            try:
                encoding = tiktoken.encoding_for_model(model_name)
                logger.info(f"model {model_name} uses tiktoken to automatically recognize encoding")
            except KeyError:
                # for models that tiktoken cannot recognize, use cl100k_base as default encoding
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 uses cl100k_base encoding
                logger.warning(
                    f"model {model_name} cannot be automatically recognized by tiktoken, use default encoding cl100k_base "
                    f"(note: this may not be accurate, it is recommended to specify it explicitly in MODEL_ENCODING)"
                )

    seen_ids = set()
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    if os.path.exists(out_file):
        with open(out_file, mode="r", encoding="utf-8") as fr:
            for line_num, line in enumerate(fr, start=1):
                line = line.strip()
                if not line:  # skip empty line
                    continue
                try:
                    data = json.loads(line)
                    seen_ids.add(data["instance_id"])
                except json.JSONDecodeError as e:
                    # show specific error position
                    error_pos = e.pos if hasattr(e, 'pos') else None
                    line_preview = line[:200] if len(line) > 200 else line  # only show first 200 characters
                    print(f"❌ JSON decode error in file: {out_file}")
                    print(f"    line number: {line_num}")
                    if error_pos is not None:
                        print(f"    error position: {error_pos}th character")
                    print(f"    error message: {str(e)}")
                    print(f"    line content preview: {line_preview}...")
                    print(f"    line total length: {len(line)} characters")
                    raise ValueError(
                        f"output file {out_file} line {line_num} JSON decode error: {str(e)}\n"
                        f"error position: {error_pos}th character\n"
                        f"line content preview: {line_preview}"
                    ) from e
    # filter conditions: 1. not processed yet 2. token length does not exceed model limit
    model_limit = MODEL_LIMITS.get(model_name, 4096)
    
    # count total number of instances before filtering
    total_count = len(inf_instances)
    
    # count instances filtered out
    in_existing_ids = []  # in existing_ids
    token_exceeded_before = []  # token length exceeds model limit (before removing files)
    token_exceeded_after = []  # token length exceeds model limit (after removing files still exceeds)
    in_seen_ids = []  # in seen_ids (already processed)
    files_removed_count = []  # count number of files removed
    token_lengths = []  # count token length information
    
    filtered_instances = []
    filtered_instances_ids = []
    
    for inf_instance in inf_instances:
        instance_id = inf_instance.instance_id
        original_token_count = gpt_tokenize(inf_instance.prompt, encoding)
        token_lengths.append(original_token_count)
        
        # check filter conditions
        skip_reasons = []
        
        if instance_id in existing_ids:
            in_existing_ids.append((instance_id, original_token_count))
            skip_reasons.append("in existing_ids")
        
        if instance_id in seen_ids:
            in_seen_ids.append((instance_id, original_token_count))
            skip_reasons.append("in seen_ids (already processed)")
        
        # if already filtered out by other conditions, skip
        if skip_reasons:
            continue
        
        # handle cases where token exceeds limit: try to reduce files
        current_token_count = original_token_count
        current_prompt = inf_instance.prompt
        removed_files = 0
        
        if current_token_count > model_limit:
            token_exceeded_before.append((instance_id, original_token_count, model_limit))
            
            # try to reduce files until the limit is satisfied
            max_removal_attempts = 20  # maximum number of attempts to reduce files (to avoid infinite loop)
            for attempt in range(max_removal_attempts):
                # reduce the last file
                new_prompt = remove_last_file_from_prompt(current_prompt)
                
                # check if there are any files that can be removed (by checking if code_context is empty)
                if new_prompt == current_prompt:
                    # cannot reduce any more files, exit loop
                    break
                
                # recalculate token length
                new_token_count = gpt_tokenize(new_prompt, encoding)
                
                if new_token_count <= model_limit:
                    # satisfies limit, use new prompt
                    current_prompt = new_prompt
                    current_token_count = new_token_count
                    removed_files = attempt + 1
                    logger.info(
                        f"instance {instance_id} removed {removed_files} files, "
                        f"token from {original_token_count} to {current_token_count}, satisfies limit {model_limit}"
                    )
                    break
                else:
                    # still exceeds limit, continue reducing
                    current_prompt = new_prompt
                    current_token_count = new_token_count
                    removed_files = attempt + 1
            
            # check if finally satisfies limit
            if current_token_count > model_limit:
                # still exceeds limit after reducing files, filter out
                token_exceeded_after.append((instance_id, original_token_count, current_token_count, model_limit, removed_files))
                skip_reasons.append(f"token length exceeds limit ({current_token_count} > {model_limit}), removed {removed_files} files still cannot satisfy")
            else:
                # satisfies limit, update prompt and keep
                inf_instance.prompt = current_prompt
                if removed_files > 0:
                    files_removed_count.append((instance_id, removed_files, original_token_count, current_token_count))
        
        # if satisfies all conditions, keep the instance
        if not skip_reasons:
            filtered_instances.append(inf_instance)
            filtered_instances_ids.append((instance_id, current_token_count))
    
    # output statistics
    print(f"\n{'='*80}")
    print(f"📊 instance filter statistics:")
    print(f"{'='*80}")
    print(f"total number of instances: {total_count}")
    print(f"valid number of instances: {len(filtered_instances)}")
    print(f"number of instances filtered out: {total_count - len(filtered_instances)}")
    
    if token_lengths:
        avg_tokens = sum(token_lengths) / len(token_lengths)
        max_tokens = max(token_lengths)
        min_tokens = min(token_lengths)
        print(f"\nToken length statistics:")
        print(f"  - average length: {avg_tokens:.1f}")
        print(f"  - maximum length: {max_tokens}")
        print(f"  - minimum length: {min_tokens}")
        print(f"  - model limit: {model_limit}")
    
    print(f"\nfilter reason classification:")
    print(f"  - in existing_ids: {len(in_existing_ids)} instances")
    if in_existing_ids:
        example_ids = [item[0] for item in in_existing_ids[:5]]
        example_tokens = [item[1] for item in in_existing_ids[:5]]
        print(f"    examples: {example_ids}{'...' if len(in_existing_ids) > 5 else ''}")
        print(f"    corresponding token length: {example_tokens}{'...' if len(in_existing_ids) > 5 else ''}")
    
    print(f"  - token length exceeds limit ({model_limit}): {len(token_exceeded_before)} instances (before reducing files)")
    if token_exceeded_before:
        example_ids = [item[0] for item in token_exceeded_before[:5]]
        example_tokens = [(item[1], item[2]) for item in token_exceeded_before[:5]]
        print(f"    examples: {example_ids}{'...' if len(token_exceeded_before) > 5 else ''}")
        print(f"    corresponding token length/limit: {example_tokens}{'...' if len(token_exceeded_before) > 5 else ''}")
        if len(token_exceeded_before) > 0:
            max_exceeded = max(item[1] for item in token_exceeded_before)
            print(f"    maximum exceeded token length: {max_exceeded}")
    
    if files_removed_count:
        print(f"\n  - after removing files, satisfies limit: {len(files_removed_count)} instances")
        example_ids = [item[0] for item in files_removed_count[:5]]
        example_info = [(item[1], item[2], item[3]) for item in files_removed_count[:5]]
        print(f"    examples: {example_ids}{'...' if len(files_removed_count) > 5 else ''}")
        print(f"    corresponding (removed files, original token, reduced token): {example_info}{'...' if len(files_removed_count) > 5 else ''}")
    
    if token_exceeded_after:
        print(f"\n  - after removing files, still exceeds limit: {len(token_exceeded_after)} instances")
        example_ids = [item[0] for item in token_exceeded_after[:5]]
        example_info = [(item[2], item[3], item[4]) for item in token_exceeded_after[:5]]
        print(f"    examples: {example_ids}{'...' if len(token_exceeded_after) > 5 else ''}")
        print(f"    corresponding (reduced token, limit, removed files): {example_info}{'...' if len(token_exceeded_after) > 5 else ''}")

    print(f"  - in seen_ids (already processed): {len(in_seen_ids)} instances")
    if in_seen_ids:
        example_ids = [item[0] for item in in_seen_ids[:5]]
        example_tokens = [item[1] for item in in_seen_ids[:5]]
        print(f"    examples: {example_ids}{'...' if len(in_seen_ids) > 5 else ''}")
        print(f"    corresponding token length: {example_tokens}{'...' if len(in_seen_ids) > 5 else ''}")
    
    print(f"{'='*80}\n")
    
    inf_instances = filtered_instances
    print(f"instances to process: ")
    inf_instances_ids = filtered_instances_ids
    print(f"inf_instances_ids: {inf_instances_ids}")
    if not inf_instances:
        print("⚠️  no instances to process")
        return


    # create output directory
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)

    # print configuration information
    print(f"📋 inference configuration:")
    print(f"   - model: {model_name}")
    print(f"   - number of threads: {max_workers}")
    print(f"   - number of API keys: {len(openai_keys)}")
    print(f"   - number of instances to process: {len(inf_instances)}")
    print(f"   - model parameters: temperature={model_args['temperature']}, top_p={model_args['top_p']}")
    print(f"   - output file: {out_file}")
    print(f"   - used API keys: {[key[:5] + '*' * 10 + key[-5:] for key in openai_keys]}")
    print(f"   - Base URL: {os.getenv('BASE_URL')}")

    # initialize worker threads
    workers = [InferenceWorker(key) for key in openai_keys]
    result_lock = threading.Lock()  # ensure thread-safe file writing
    total_cost = 0.0
    cost_lock = threading.Lock()  # ensure thread-safe cost calculation

    def process_instance(datum, worker_idx: int):
        """process a single instance, assign to the specified worker"""
        nonlocal total_cost
        worker = workers[worker_idx % len(workers)]  # round-robin allocation of keys
        cost = worker.run_inference(datum, model_name, model_args, result_lock, out_file, timeout=timeout)

        # thread-safe update total cost
        with cost_lock:
            nonlocal total_cost
            total_cost += cost
            masked_key = worker.api_key[:5] + '*' * 10 + worker.api_key[-5:]
            print(
                f"💰 total cost: {total_cost:.2f} $ (current key: {masked_key}, instance: {datum.instance_id}, cost: {cost:.4f} $)")

    # use thread pool to execute tasks
    print("\n🚀 start multi-threaded inference...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # create task list, assign worker index to each instance
        tasks = [
            executor.submit(process_instance, datum, idx)
            for idx, datum in enumerate(inf_instances)
        ]

        # show progress bar
        for _ in tqdm(concurrent.futures.as_completed(tasks),
                      total=len(tasks), desc=f"Inference for {model_name}"):
            pass

    # output final statistics
    print("\n🎉 inference completed!")
    print(f"📊 statistics:")
    print(f"   - total number of instances processed: {len(inf_instances)}")
    # print(f"   - number of successful instances: {sum(1 for task in tasks if task.result() is not None and task.result() > 0)}")
    print(f"   - total cost: {total_cost:.2f} $")
    print(f"   - output file: {out_file}")


def main(
    dataset_names: List[str],
    model_names: List[str],
    run_ids: List[str],
    sampled_ids_file: Optional[Path] = None,
    max_workers: int = 10,
    timeout: int = 600,
) -> None:
    """
    Run inference over (dataset_name × model × run_id) combinations.
    Input paths: {SRC_INF_BENCHMARK_DATA}/{name}-task-instances_{run_id}.jsonl
    """
    sampled_instance_ids = None
    if sampled_ids_file and Path(sampled_ids_file).exists():
        try:
            with open(sampled_ids_file, "r", encoding="utf-8") as f:
                sampled_data = json.load(f)
                sampled_instance_ids = sampled_data.get("sampled_instance_ids", [])
                print(f"📋 read sampled file: {sampled_ids_file}")
                print(f"    number of sampled instances: {len(sampled_instance_ids)}")
        except Exception as e:
            print(f"⚠️  read sampled file failed: {e}, will process all instances")
            sampled_instance_ids = None
    else:
        if sampled_ids_file:
            print(f"ℹ️  sampled file not found: {sampled_ids_file}, will process all instances")
        else:
            print(f"ℹ️  no sampled file specified, will process all instances")

    for name in dataset_names:
        for model_name in model_names:
            for run_id in run_ids:
                model_args = MAP_MODEL_TO_COFIG[model_name]
                src = Path(
                    f"{SRC_INF_BENCHMARK_DATA}/{name.replace('/', '-')}-task-instances_{run_id}.jsonl"
                )
                out_file = (
                    f"{EXPERIMENTAL_RESULTS / model_name}"
                    f"/T={model_args['temperature']}/n={model_args['n']}/{src.name}"
                )
                print("\n" + "=" * 80)
                print("🚀 start executing tasks:")
                print(f"   - dataset: {name}")
                print(f"   - model: {model_name}")
                print(f"   - run ID: {run_id}")
                print(f"   - input file: {src}")
                print(f"   - output file: {out_file}")
                print(f"   - model parameters: temperature={model_args['temperature']}, top_p={model_args.get('top_p', 'N/A')}, n={model_args['n']}")
                print(f"   - timeout: {timeout} seconds")
                if sampled_instance_ids:
                    print(f"   - using sampled instances: {len(sampled_instance_ids)} instances")
                print("=" * 80 + "\n")

                inference_openai(
                    src,
                    model_name,
                    model_args,
                    out_file,
                    max_workers=max_workers,
                    timeout=timeout,
                    target_instance_ids=sampled_instance_ids,
                )


if __name__ == "__main__":
    # Example:
    #   python -m editbench.inference.run_api --dataset-name all --model deepseek-v3.2 --run-id 0.2
    #   python -m editbench.inference.run_api --dataset-name all --model qwen3-235b-a22b deepseek-v3.2 --run-id 0.2_bm25_1 0.2_bm25_5 --max-workers 10 --timeout 600
    #   python -m editbench.inference.run_api --dataset-name all --model claude-sonnet-4-5-20250929 --run-id 0.2 --sampled-ids-file ./crawled_data/infbench/sampled_instance_ids_0.2.json
    parser = argparse.ArgumentParser(
        description="Run API inference over infbench datasets (dataset × model × run_id).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        nargs="+",
        default=["all"],
        help="Dataset name(s), e.g. all or django/django (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="Model name(s), e.g. deepseek-v3.2 claude-sonnet-4-5-20250929.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        nargs="+",
        required=True,
        help="Run ID(s), e.g. 0.2 or 0.2_bm25_1 0.2_bm25_5.",
    )
    parser.add_argument(
        "--sampled-ids-file",
        type=str,
        default=None,
        help="Optional JSON file with sampled_instance_ids to restrict instances.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max concurrent workers (default: 10).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per request in seconds (default: 600).",
    )
    args = parser.parse_args()
    main(
        dataset_names=args.dataset_name,
        model_names=args.model,
        run_ids=args.run_id,
        sampled_ids_file=Path(args.sampled_ids_file) if args.sampled_ids_file else None,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )
