from typing import List, Union, Optional, Literal
import dataclasses

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

model_path = "/scratch/st-fhendija-1/nikta/deep_model"

class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError
    
class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        import torch
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.tolist()  # Convert tensor to list of token IDs

        # Decode outputs
        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        # Ensure `outs` is a list of strings
        assert isinstance(outs, list), f"Expected outs to be a list, got {type(outs)}"
        for i, out in enumerate(outs):
            assert isinstance(out, str), f"Expected out to be a string, got {type(out)}"

        # If you need additional processing, call extract_output only if necessary
        # Assuming extract_output applies further processing on decoded strings:
        outs = [self.extract_output(out) for out in outs]

        # Return the decoded output(s)
        if len(outs) == 1:
            return outs[0]  # Return a single string
        else:
            return outs  # Return a list of strings

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class DeepSeekCoder(HFModelBase):
    import torch
    def __init__(self, model_path=None):
        from typing import List, Union
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        """
        Initialize the DeepseekCoder model.

        :param model_path: local path to the model if you have downloaded it locally.
        """

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path is not None else "deepseek-ai/deepseek-coder-6.7b-instruct",
            quantization_config=nf4_config,
            trust_remote_code=True,
            torch_dtype="auto"
        )


        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path is not None else f"deepseek-ai/deepseek-coder-6.7b-instruct",
            trust_remote_code=True
        )

        super().__init__("deepseek-ai/deepseek-coder-6.7b-instruct", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):

        bos_token = "<|begin▁of▁sentence|>"
        eos_token = "<|end▁of▁sentence|>"
        prompt = bos_token

        for message in messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}{eos_token}"
            elif message.role == "system":
                prompt += f"{message.content}\n\n"

        prompt += "Assistant:"
        
        return self.tokenizer.encode(prompt, return_tensors="pt")

    def extract_output(self, output: str) -> str:

        out = output.split("Assistant: ", 1)[-1]  # Get everything after "Assistant: "
        eos_token = "<｜end▁of▁sentence｜>"
        if out.endswith(eos_token):
            out = out[:-len(eos_token)]  # Remove EOS token if present
        return out.strip()