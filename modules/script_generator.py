import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import config
from modules.token_counter import TokenCounter
import json
import os

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

def generate_tiktok_script(content: dict, target_audience: str, token_counter: TokenCounter) -> str:
    prompt = f"""{HUMAN_PROMPT} Create an engaging TikTok script based on the following content and guidelines:
    
    Content Summary: {content['summary']}
    Key Facts: {', '.join(content['key_facts'])}
    Hooks: {', '.join(content['hooks'])}
    Simplified Content: {content['simplified_content']}
    Anecdote: {content['anecdote']}
    Target Audience: {target_audience}
    
    Guidelines:
    1. Start with a strong hook that grabs attention immediately.
    2. Keep the script short and simple, suitable for a 60-second video.
    3. Use natural, conversational language as if speaking to a friend.
    4. Tailor the content and language to the target audience.
    5. Include at least one key fact or interesting tidbit.
    6. End with a memorable conclusion or call-to-action.{AI_PROMPT} Here's an engaging TikTok script based on the provided content and guidelines:

    """

    token_counter.add_prompt_tokens(len(prompt.split()))  # Simple word count as a token estimate

    try:
        response = client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=300,
            stop_sequences=[HUMAN_PROMPT],
        )
        script = response.completion
    except Exception as e:
        raise Exception(f"Error in script generation: {str(e)}")

    token_counter.add_completion_tokens(len(script.split()))  # Simple word count as a token estimate

    return script

def analyze_script_engagement(script: str, token_counter: TokenCounter) -> dict:
    prompt = f"""{HUMAN_PROMPT} Analyze the following TikTok script for potential engagement factors:
    
    Script: {script}
    
    Please provide ratings (1-10) and brief explanations for the following aspects:
    1. Hook Strength
    2. Clarity and Simplicity
    3. Audience Appropriateness
    4. Information Value
    5. Entertainment Factor
    6. Call-to-Action Effectiveness{AI_PROMPT} Here's my analysis of the TikTok script's engagement factors:

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
        raise Exception(f"Error in script engagement analysis: {str(e)}")

    token_counter.add_completion_tokens(len(analysis.split()))  # Simple word count as a token estimate

    return {"engagement_analysis": analysis}

def save_script(script: str, metadata: dict, filename: str):
    output = {
        "script": script,
        "metadata": metadata
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Script saved to {filename}")
    except Exception as e:
        print(f"Error saving script: {str(e)}")