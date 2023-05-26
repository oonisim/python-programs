"""
OpenAPI Chat API module
[References]
    https://platform.openai.com/docs/api-reference
    https://platform.openai.com/docs/api-reference/chat/create
    https://github.com/openai/openai-cookbook

[Models]
Available models for the endpoint is listed in:
https://platform.openai.com/docs/models/model-endpoint-compatibility
| ENDPOINT             | MODEL NAME    (As of MAR/2023)                                                  |
|----------------------|---------------------------------------------------------------------------------|
| /v1/chat/completions | gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301 |

[Request Limits]
OpenAI API has separate limits for requests per minute and tokens per minute.

- openai.error.RateLimitError
429: 'Too Many Requests' when the request rate exceeded the limit.
https://platform.openai.com/docs/guides/rate-limits/overview
https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb

Throttling parallel requests to avoid rate limit errors:
https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

[Batching requests]
https://platform.openai.com/docs/guides/rate-limits/batching-requests
"""
import os
import re
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
)

from util_python.function import (  # pylint: disable=import-error
    retry_with_exponential_backoff
)

import openai


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "text-davinci-003"
]
OPENAI_MODEL_CHAT_COMPLETIONS = "gpt-3.5-turbo"
OPENAI_MODEL_TEXT_COMPLETIONS = "text-davinci-003"
OPENAI_ERRORS_TO_RETRY: Tuple = (
    openai.error.RateLimitError,
    openai.error.APIConnectionError,
    openai.error.APIError
)


# --------------------------------------------------------------------------------
# API class
# --------------------------------------------------------------------------------
class OpenAI:
    """OpenAI API implementation class"""
    # --------------------------------------------------------------------------------
    # Static
    # --------------------------------------------------------------------------------
    @staticmethod
    def list_models():
        """List Open AI Models"""
        return openai.Model.list()

    # --------------------------------------------------------------------------------
    # Instance
    # --------------------------------------------------------------------------------
    def __init__(
            self,
            path_to_api_key: str,
            model_chat_completions: str = OPENAI_MODEL_CHAT_COMPLETIONS,
            model_text_completions: str = OPENAI_MODEL_TEXT_COMPLETIONS
    ):
        with open(file=path_to_api_key, encoding='utf-8') as api_key:
            openai.api_key = api_key.readline().strip()

        self._model_chat_completions: str = model_chat_completions
        self._model_text_completions: str = model_text_completions

    @property
    def model_chat_completions(self):
        """Open AI Model name for the chat completion task"""
        return self._model_chat_completions

    @property
    def model_text_completions(self):
        """Open AI Model name for the text completion task"""
        return self._model_text_completions

    @retry_with_exponential_backoff(
        proactive_delay=1.0,
        errors=OPENAI_ERRORS_TO_RETRY
    )
    def get_chat_completion_by_prompt(
            self,
            prompt,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            stream=False,
    ) -> str:
        """Send a chat completion request as a prompt
        https://platform.openai.com/docs/api-reference/completions/create

        Prompt examples using the news:
        https://www.sbs.com.au/news/article/french-government-survives-no-confidence-votes-in-pension-fight/rmgr77do1

        "Summarize the text. Must be less than 15 words. Text=<text>"
        > French government survives no-confidence motions, but faces pressure over controversial pensions reform.

        "Notable figures: <text>"
        > French President Emmanuel Macron,
        > Prime Minister Elisabeth Borne,
        > Finance Minister Bruno Le Maire,
        > and Republican MP Aurelien Pradie.

        Args:
            prompt: A task instruction to the model
            temperature: randomness of the generated responses
            max_tokens: The maximum number of tokens to generate in the chat completion.
            stream: if set, partial message deltas will be sent, like in ChatGPT
        Returns: Completion
        """
        response = openai.ChatCompletion.create(
            model=self.model_chat_completions,
            messages=[
                {"role": "assistant", "content": "You are succinct and simple."},
                {"role": "user", "content": re.sub(r"['\"/\s]+", ' ', prompt, flags=re.MULTILINE)}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        return response['choices'][0]['message']['content']

    @retry_with_exponential_backoff(
        proactive_delay=1.0,
        errors=OPENAI_ERRORS_TO_RETRY
    )
    def get_chat_completion_by_messages(
            self, messages: List[Dict[str, str]]
    ) -> str:
        """Send a chat completion request as a message
        https://platform.openai.com/docs/api-reference/completions/create

        Message example:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]

        Args:
            messages:
                messages as in the expected format by API.
                See https://platform.openai.com/docs/guides/chat/introduction.
        Returns: Completion
        """
        response = openai.ChatCompletion.create(
            model=self.model_chat_completions,
            messages=messages
        )
        return response['choices'][0]['message']['content']


def run_get_chat_completion_by_prompt(prompt):
    """Test"""
    open_ai = OpenAI(
        path_to_api_key=f"{os.path.expanduser('~')}/.openai/api_key"
    )
    print(open_ai.get_chat_completion_by_prompt(prompt=prompt))


if __name__ == "__main__":
    TEXT = """美国警方称，边界巡防队在德州南部圣安东尼奥市（San Antonio）附近拦下一列有无证移民乘坐的货运列车火车，
其中数十人需要就医，两人已经死亡。这起悲剧发生在2022一起更严重的窒息死亡事件现场的附近，当时天气闷热，
有53名移民在一次偷渡活动中在一辆拖拉机拖车的后面死亡。德克萨斯州乌弗拉德镇的官员接到了一个匿名紧急电话，
告知他们有许多移民在一列火车内呼吸困难。美国国土安全部的调查人员表示，他们现在正在调查人口走私的可能性。
"""
    run_get_chat_completion_by_prompt(
        "Summarize the TEXT less than 20 words. Response be less than 20 words. TEXT=" + TEXT
    )
