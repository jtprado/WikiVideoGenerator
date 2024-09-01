from typing import Dict, List
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import config
from modules.token_counter import TokenCounter
from modules.embeddings import EmbeddingModel

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
embedding_model = EmbeddingModel()

def analyze_content(content: str, query: str, audience: str, token_counter: TokenCounter) -> Dict:
    prompt = f"""{HUMAN_PROMPT} Analyze the following content about '{query}' for a TikTok video targeted at {audience}:

    Content: {content}

    Provide the following:
    1. A brief summary (2-3 sentences)
    2. 5 key facts
    3. 3 attention-grabbing hooks
    4. A simplified explanation of any complex concepts
    5. A short, relatable anecdote or example{AI_PROMPT} Here's my analysis:

    """

    token_counter.add_prompt_tokens(len(prompt.split()))  # Simple word count as a token estimate

    try:
        response = client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=500,
            stop_sequences=[HUMAN_PROMPT],
        )
        analysis = response.completion
    except Exception as e:
        raise Exception(f"Error in content analysis: {str(e)}")

    token_counter.add_completion_tokens(len(analysis.split()))  # Simple word count as a token estimate

    # Parse the analysis into structured data
    lines = analysis.split('\n')
    parsed_analysis = {
        'summary': '',
        'key_facts': [],
        'hooks': [],
        'simplified_content': '',
        'anecdote': ''
    }

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('1.') and 'summary' in line.lower():
            current_section = 'summary'
        elif line.startswith('2.') and 'key facts' in line.lower():
            current_section = 'key_facts'
        elif line.startswith('3.') and 'hooks' in line.lower():
            current_section = 'hooks'
        elif line.startswith('4.') and 'simplified' in line.lower():
            current_section = 'simplified_content'
        elif line.startswith('5.') and 'anecdote' in line.lower():
            current_section = 'anecdote'
        elif current_section:
            if current_section in ['summary', 'simplified_content', 'anecdote']:
                parsed_analysis[current_section] += line + ' '
            else:
                parsed_analysis[current_section].append(line)

    return parsed_analysis

def query_index(indexer, query: str, token_counter: TokenCounter) -> List[Dict]:
    query_embedding = embedding_model.embed([query])[0]
    token_counter.add_embedding_tokens(len(query.split()))  # Simple word count as a token estimate
    return indexer.search_index(query_embedding)