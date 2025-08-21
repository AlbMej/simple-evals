import time
from typing import Any

import openai
from openai import OpenAI

from ..eval_types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        max_retries: int = 5,  # A limit for retries
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while trial < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tool_choice="none",           # avoid tool calls, content won’t be None
                    timeout=60,                   # avoid hanging forever
                )
                msg = response.choices[0].message
                content = msg.content
                # If provider still returns None content, treat as a skip rather than infinite retry
                if content is None:
                    # log tool_calls for debugging
                    tc = getattr(msg, "tool_calls", None)
                    print("Warning: model returned tool call or empty content; skipping. tool_calls=", tc)
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": response.usage},
                        actual_queried_message_list=message_list,
                        )
                # If we succeed, return immediately
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )

            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                trial += 1
                if trial >= self.max_retries:
                    # If we've run out of retries, raise the final error
                    print(f"Failed after {self.max_retries} retries. Giving up on this question.")
                    raise RuntimeError(f"API call failed after all retries: {e}") from e
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Caught error: {e}. Retrying ({trial}/{self.max_retries}) after {exception_backoff} sec..."
                )
                time.sleep(exception_backoff)
            # unknown error shall throw exception
