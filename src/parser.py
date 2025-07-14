import os
import re
import time
import json
import requests
import wikipediaapi
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path

class RussianTextParser:
    def __init__(self, raw_dir, parsed_dir, min_text_length=10000):
        self.raw_dir = raw_dir
        self.parsed_dir = parsed_dir
        self.min_text_length = min_text_length

    def parse_wikipedia(self, topic: str, lang: str = "ru") -> str:
        wiki = wikipediaapi.Wikipedia(
            user_agent="MyResearchBot/1.0",
            language=lang,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        page = wiki.page(topic)
        if not page.exists():
            return ""
        
        text = page.text
        for section in ["См. также", "Примечания", "Литература", "Ссылки"]:
            text = text.split(f"\n\n{section}")[0]
        
        return text

    def parse_lenta_rss(self) -> list:
        url = "https://lenta.ru/rss/news"
        feed = feedparser.parse(url)
        articles = []
        
        for entry in feed.entries:
            try:
                response = requests.get(entry.link, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                body = soup.find('div', {'itemprop': 'articleBody'})
                if body:
                    text = ' '.join(p.get_text() for p in body.find_all('p'))
                    articles.append({
                        'title': entry.title,
                        'text': text,
                        'source': entry.link
                    })
            except Exception as e:
                print(f"Ошибка при парсинге {entry.link}: {str(e)}")
        
        return articles

    def parse_lib_ru(self, author: str) -> str:
        base_url = "http://az.lib.ru"
        search_url = f"{base_url}/cgi-bin/ask?key={author}&type=author"
        full_text = ""
        
        try:
            response = requests.get(search_url, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            works = []
            for li in soup.find_all('li'):
                a_tag = li.find('a')
                if a_tag and 'href' in a_tag.attrs:
                    works.append(a_tag['href'])
            
            for work in works[:3]:
                try:
                    work_url = f"{base_url}{work}"
                    work_resp = requests.get(work_url, timeout=15)
                    work_soup = BeautifulSoup(work_resp.text, 'html.parser')
                    for pre in work_soup.find_all('pre'):
                        full_text += pre.get_text() + "\n\n"
                except Exception:
                    continue
        except Exception as e:
            print(f"Ошибка при поиске автора: {str(e)}")
        
        return full_text

    def save_text(self, text: str, source: str):
        if len(text) < self.min_text_length:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{timestamp}.txt"
        filepath = self.parsed_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return True

    def run_parsing(self):
        print("\nЗапуск автоматического парсинга...")
        
        # Википедия
        wiki_topics = ["Россия", "Литература", "Наука", "История"] * 10
        for topic in wiki_topics:
            text = self.parse_wikipedia(topic)
            if text and self.save_text(text, "wikipedia"):
                print(f"Спарсено с Wikipedia: {topic} ({len(text)} символов)")
        
        # Lenta.ru
        articles = self.parse_lenta_rss()
        for article in articles:
            if self.save_text(article['text'], "lenta"):
                print(f"Спарсено новостей: {article['title']}")
        
        # Lib.ru
        authors = ["Толстой Л.Н.", "Достоевский Ф.М.", "Чехов А.П."]
        for author in authors:
            text = self.parse_lib_ru(author)
            if text and self.save_text(text, "lib_ru"):
                print(f"Спарсено произведений: {author} ({len(text)} символов)")
        
        print("Парсинг завершен!\n")