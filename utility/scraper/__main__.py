from utility.scraper.url_list import urls
from utility.scraper.pdf_generator import PDFGen


def main():
    """Main script for scraper module."""

    pdf = PDFGen(urls)
    for title, text in zip(pdf.titles, pdf.texts):
        pdf.create_pdf(title, text)


if __name__ == "__main__":
    main()