import tiktoken

class TokenCounter:
    def __init__(self):
        self.total_embedding_token_count = 0
        self.prompt_llm_token_count = 0
        self.completion_llm_token_count = 0

    @property
    def total_llm_token_count(self):
        return self.prompt_llm_token_count + self.completion_llm_token_count

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))

    def add_embedding_tokens(self, token_count: int):
        self.total_embedding_token_count += token_count

    def add_prompt_tokens(self, token_count: int):
        self.prompt_llm_token_count += token_count

    def add_completion_tokens(self, token_count: int):
        self.completion_llm_token_count += token_count

def print_token_count(token_counter, embed_model, model="gpt-3.5-turbo"):
    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )
    pricing = {
        'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
        'gpt-3.5-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
        'gpt-4-0613': {'prompt': 0.03, 'completion': 0.06},
        'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
        'embedding': {'hugging_face': 0, 'text-embedding-ada-002': 0.0001}
    }
    print(
        "Embedding Cost: ",
        pricing['embedding'][embed_model] * token_counter.total_embedding_token_count/1000,
        "\n",
        "LLM Prompt Cost: ",
        pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000,
        "\n",
        "LLM Completion Cost: ",
        pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
        "\n",
        "Total LLM Cost: ",
        pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000 + pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
        "\n",
        "Total cost: ",
        pricing['embedding'][embed_model] * token_counter.total_embedding_token_count/1000 + pricing[model]["prompt"] * token_counter.prompt_llm_token_count/1000 + pricing[model]["completion"] * token_counter.completion_llm_token_count/1000,
    )