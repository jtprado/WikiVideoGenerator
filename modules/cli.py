import click
import logging

@click.command()
@click.option('--topic', prompt='Enter a Wikipedia topic', help='The topic to fetch from Wikipedia')
@click.option('--date', prompt='Enter a date (YYYY-MM-DD)', help='The date for historical events')
@click.option('--audience', prompt='Enter target audience', help='The target audience for the TikTok script (e.g., "teens 13-17", "young adults 18-25")')
@click.option('--output-dir', default='output', help='Directory to save the generated scripts')
def get_user_input(topic, date, audience, output_dir):
    """Command line interface for the Wikipedia TikTok Generator."""
    logging.debug(f"get_user_input called with: topic={topic}, date={date}, audience={audience}, output_dir={output_dir}")
    return {
        'topic': topic,
        'date': date,
        'audience': audience,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    get_user_input()