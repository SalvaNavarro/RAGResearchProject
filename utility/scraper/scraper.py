import requests
from bs4 import BeautifulSoup, Tag


class Scraper:
    """
    Extracts the text data from the set of PDF files.

    Attributes:
        url (str): URL to scrape.
        soup (BeautifulSoup): The scraped soup object.
        main (Tag): The scraped soup object.
        title (str): The article title.
        text (str): The article main text.
        hero_image (str): The article main image file name.
        images (str): The article images file names.
    """

    def __init__(self, url) -> None:
        """
        Constructor.
        """
        self.url = url
        self.soup = self._get_soup()
        self.main = self._get_main_content()
        self.title = self._get_title()
        self.text = self._get_clean_text()
        self.hero_image = self._get_hero_image_src()
        self.images = self._get_images_src()

    def _get_soup(self) -> BeautifulSoup:
        """
        Retrieves the soup from the URL.

        Returns:
            BeautifulSoup
        """
        html = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(html.text, "html.parser")
        return soup

    def _get_main_raw(self) -> Tag:
        """
        Retrieves the main content tag from the URL.

        Returns:
            Tag
        """
        main_raw = self.soup.section
        return main_raw


    def _get_header_raw(self) -> Tag:
        """
        Retrieves the header tag from the soup.

        Returns:
            Tag
        """
        header_raw = self.soup.select_one("article header")
        return header_raw

    def _get_title(self) -> str:
        """
        Retrieves the article title from the soup.

        Returns:
            str
        """
        header = self._get_header_raw()
        title = header.select_one("h1").get_text(strip=True)

        if title is None:
            print("Title not found")
            return None

        return title


    def _get_hero_image_src(self) -> str | None:
        """
        Retrieves the article title from the soup.

        Returns:
            str
        """
        header = self._get_header_raw()
        hero_image = header.select_one("img")
        hero_image_src = hero_image["src"]

        if hero_image_src is None:
            print("Hero image not found")
            return None

        return hero_image_src


    def _get_main_content(self) -> Tag:
        """
        Retrieves the main content from the soup.

        Returns:
            Tag
        """
        main = self._get_main_raw()
        for aside in main.find_all("aside"):
            aside.decompose()

        for read_more in main.select("div.read-more"):
            read_more.decompose()

        return main


    def _get_images_src(self) -> list[str]:
        """
        Retrieves the image sources from the main content.

        Returns:
            list[str]
        """
        main = self._get_main_raw()
        images_src = []
        for img in main.find_all("img"):
            images_src.append(img["src"])

        return images_src


    def _get_clean_text(self) -> str:
        """
        Retrieves the image sources from the main content.

        Returns:
            str
        """
        text = self.main.get_text()
        clean_lines = []
        for line in text.splitlines():
            if line.strip():
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)

        return clean_text


