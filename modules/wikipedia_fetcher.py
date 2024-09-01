import wikipediaapi
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WikipediaFetcher:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia('Weisheit93/0.1 (jtprado@protonmail.com)', 'en')

    def fetch_wikipedia_content(self, topic: str) -> dict:
        """
        Fetch Wikipedia content for a given topic.

        Args:
            topic (str): The topic to fetch from Wikipedia.

        Returns:
            Dict: A dictionary containing the page title, summary, full content, and metadata.
        """
        page = self.wiki_wiki.page(topic)
        
        if not page.exists():
            raise ValueError(f"No Wikipedia page found for '{topic}'.")
        
        return {
            'title': page.title,
            'summary': page.summary,
            'content': self.format_wikipedia_content(page),
            'url': page.fullurl,
            'retrieval_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def format_wikipedia_content(self, page: wikipediaapi.WikipediaPage) -> str:
        def format_sections(sections, level=0):
            formatted_content = ""
            for s in sections:
                formatted_content += f"{'#' * (level + 1)} {s.title}\n\n{s.text}\n\n"
                formatted_content += format_sections(s.sections, level + 1)
            return formatted_content

        return format_sections(page.sections)

    def save_content(self, content: dict, filename: str) -> None:
        """
        Save content to a JSON file.

        Args:
            content (Dict): The content to save.
            filename (str): The filename to save the content to.
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            logger.info(f"Content saved to {filename}")
        except IOError as e:
            logger.error(f"Error saving content to {filename}: {e}")

    def fetch_and_save_wikipedia_content(self, topic: str, output_dir: str) -> str:
        """
        Fetch Wikipedia content for a topic and save it to a file.

        Args:
            topic (str): The topic to fetch from Wikipedia.
            output_dir (str): The directory to save the content to.

        Returns:
            str: The filename where the content was saved.
        """
        safe_filename = "".join(c if c.isalnum() else "_" for c in topic)
        filename = os.path.join(output_dir, f"{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        content = self.fetch_wikipedia_content(topic)
        self.save_content(content, filename)
        logger.info(f'Content for {topic} fetched and saved to {filename}')
        return filename

    def load_content(self, filename: str) -> dict:
        """
        Load content from a JSON file.

        Args:
            filename (str): The filename to load the content from.

        Returns:
            Dict: The loaded content.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except IOError as e:
            logger.error(f"Error loading content from {filename}: {e}")
            return {}