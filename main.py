import os
import sys
import time
import psutil
import traceback

from modules.wikipedia_fetcher import WikipediaFetcher
from modules import content_indexer, content_analyzer, script_generator, cli, token_counter
from modules.token_counter import TokenCounter, print_token_count
from modules.embeddings import EmbeddingModel

import requests
import argparse
import config
from openai import OpenAI
import anthropic

# Check OpenAI API key
if not config.OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")

# Check Anthropic API key
if not config.ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key is missing. Please set the ANTHROPIC_API_KEY environment variable.")

def check_output_directory(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except PermissionError:
            raise PermissionError(f"Unable to create output directory: {output_dir}. Please check permissions.")
    elif not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}. Please check permissions.")

def debug_get_user_input():
    print("Debug: Entering get_user_input function")
    try:
        topic = input("Enter a Wikipedia topic: ")
        print(f"Debug: Received topic: {topic}")
        audience = input("Enter target audience: ")
        print(f"Debug: Received audience: {audience}")
        output_dir = input("Enter output directory (default: output): ") or "output"
        print(f"Debug: Received output_dir: {output_dir}")
        return {
            'topic': topic,
            'audience': audience,
            'output_dir': output_dir
        }
    except Exception as e:
        print(f"Debug: Exception in get_user_input: {str(e)}")
        raise

def main():
    start_time = time.time()
    max_execution_time = 300  # 5 minutes

    try:
        print('Starting Wikipedia TikTok Generator')
        print("Preparing to get user input...")
        try:
            user_input = debug_get_user_input()
            print("User input received successfully.")
        except Exception as e:
            print(f"Error in get_user_input: {str(e)}")
            raise

        topic = user_input['topic']
        audience = user_input['audience']
        output_dir = user_input['output_dir']

        print(f"Received user input - Topic: {topic}, Audience: {audience}")

        print("Checking output directory...")
        check_output_directory(output_dir)
        print("Output directory check completed.")

        # Create necessary directories
        content_dir = os.path.join(output_dir, 'content')
        os.makedirs(content_dir, exist_ok=True)
        script_dir = os.path.join(output_dir, 'scripts')
        os.makedirs(script_dir, exist_ok=True)

        # Initialize token counter and embedding model
        token_counter = TokenCounter()
        embedding_model = EmbeddingModel()

        # Fetch Wikipedia content
        print(f'Fetching Wikipedia content for topic: {topic}')
        wikipedia_fetcher = WikipediaFetcher()
        topic_content = wikipedia_fetcher.fetch_wikipedia_content(topic)
        topic_filename = os.path.join(content_dir, f"{topic.replace(' ', '_')}.json")
        wikipedia_fetcher.save_content(topic_content, topic_filename)
        print(f'Wikipedia content saved to: {topic_filename}')

        # Create or load index
        print("Creating or loading content index...")
        index = content_indexer.load_index(token_counter)
        index.add_document_to_index(topic_content)
        print('Content indexing completed')

        # Analyze content
        print('Analyzing content')
        analysis = content_analyzer.analyze_content(topic_content['content'], topic, audience, token_counter)
        print('Content analysis completed')

        # Generate TikTok script
        print('Generating TikTok script')
        script = script_generator.generate_tiktok_script(analysis, audience, token_counter)
        print('TikTok script generated')

        # Save script
        print("Saving script...")
        script_filename = os.path.join(script_dir, f"{topic.replace(' ', '_')}_script.txt")
        try:
            with open(script_filename, 'w', encoding='utf-8') as f:
                f.write(script)
            print(f'Script saved to {script_filename}')
        except Exception as e:
            print(f"Error saving script: {str(e)}")

        # Analyze script engagement
        print("Analyzing script engagement...")
        engagement_analysis = script_generator.analyze_script_engagement(script, token_counter)

        print(f"\nScript generated successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Script file: {script_filename}")
        print("\nEngagement Analysis:")
        print(engagement_analysis['engagement_analysis'])

        # Print token count and cost
        print("\nToken Usage and Cost:")
        print_token_count(token_counter, config.EMBEDDING_MODEL, config.LLM_MODEL)

        print('Wikipedia TikTok Generator completed successfully')

    except requests.exceptions.RequestException as e:
        print(f"A network error occurred. Please check your internet connection and try again.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except PermissionError as e:
        print(f"Permission error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        if execution_time >= max_execution_time:
            print("Script execution took longer than expected. Please check the log file for details.")
        
        # Ensure all logs are written
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wikipedia TikTok Generator')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    main()