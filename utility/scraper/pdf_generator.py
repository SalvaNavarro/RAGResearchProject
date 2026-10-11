import os
import requests
from io import BytesIO
import re

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet

from utility.scraper.scraper import Scraper
from config import DB_SETUP_DATA, g_logger


class PDFGen:
    """
    Generates a set of clean PDF files from scraped data.

    Attributes:
        urls (list[str]): A list of URLs to scrape.
        titles (list[str]): A list of titles.
        texts (list[str]): A list of body texts.
        images (list[str]): A list of the images on the article.
    """

    def __init__(self, urls) -> None:
        """
        Constructor.
        """
        self.urls = urls
        self.titles = []
        self.texts = []
        self.images = []
        self._get_info()


    def _scrape(self) -> list[Scraper]:
        """
        Creates a list of Scraper objects from the given urls.

        Returns:
            list[Scraper]
        """
        scrapers = []
        for url in self.urls:
            try:
                scraper = Scraper(url)
                scrapers.append(scraper)
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        return scrapers
    
    def _get_info(self) -> None:
        """
        Populates titles, texts and images lists.

        Returns:
            None
        """
        scrapers = self._scrape()
        for scrape in scrapers:
            self.titles.append(scrape.title)
            self.texts.append(scrape.text)
            img = []
            img.append(scrape.hero_image)
            for image in scrape.images:
                img.append(image)

            self.images.append(img)


    def create_pdf(self, title, text) -> None:
        """
        Creates the PDF files.

        Returns:
            None
        """

        # start index and image URLs
        index = self.titles.index(title)
        img_urls = self.images[index]
        file_name = self._create_filename(title)

        # set up canvas and frames
        pagesize = A4
        width, height = pagesize
        canvas, cover_frame, page_frame, styles = self._format_canvas(file_name, pagesize)

        # creating title and body texts
        title_text = Paragraph(title, styles["Title"])
        body_text = Paragraph(text, styles["BodyText"])
        
        # add title to front page
        cover_frame.addFromList([title_text], canvas)
        canvas.showPage()

        # format text (summary) page
        page_frame.addFromList([body_text], canvas)
        canvas.showPage()

        # image page creation (simple layout with one image per page)
        for url in img_urls:

            # request img from url with raise for status if failed
            resp = requests.get(url)
            resp.raise_for_status()
            # buffer image and create reportlab ImageReader instance
            buf = BytesIO(resp.content)
            img = ImageReader(buf)

            # scale image to fit
            img_width, img_height = img.getSize()                  
            max_width, max_height = width - 2*72, height - 2*72            
            aspect = img_height / img_width

            # fit and center images
            draw_width = min(img_width, max_width)
            draw_height = draw_width * aspect

            if draw_height > max_height:
                draw_height = max_height
                draw_width = draw_height / aspect

            # center images within layout
            x = (width - draw_width) / 2
            y = (height - draw_height) / 2

            # center images within layout
            canvas.drawImage(img, x, y, draw_width, draw_height, 
                             preserveAspectRatio=True)
            canvas.showPage()

        # save file
        canvas.save()
        path = os.path.abspath(file_name)

        # log activity
        g_logger.info(f"PDF saved to: {str(path)}")


    @staticmethod
    def _create_filename(title) -> str:
        """
        Helper method to create filenames for scraped PDFs.

        Returns:
            str
        """

        clean_title = re.sub(r'[^A-Za-z0-9]', '', title)
        file_path = DB_SETUP_DATA / f"{clean_title}.pdf"
        file_name = str(file_path)

        return file_name

    @staticmethod
    def _format_canvas(file_name, 
        pagesize) -> tuple[canvas.Canvas, Frame, Frame, tuple[float, float]]:
        """
        Helper method to format canvases for PDF layout.

        Returns:
            tuple[canvas.Canvas, Frame, Frame, tuple[float, float]]
        """

        # creating a canvas object
        c = canvas.Canvas(file_name, pagesize=pagesize)
        width, height = pagesize
        styles = getSampleStyleSheet()

        # region that will contain the cover content. 72 points per inch
        cover_frame = Frame(
            x1=72, 
            y1=72, 
            width=width-2*72, 
            height=height/2, 
            showBoundary=0
            )
        # region that will contain the page content. 72 points per inch
        page_frame = Frame(
            x1=72, 
            y1=72, 
            width=width-2*72, 
            height=height-2*72, 
            showBoundary=0
        )

        return c, cover_frame, page_frame, styles