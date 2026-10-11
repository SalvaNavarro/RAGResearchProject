import os
import pymupdf
from pathlib import Path
from config import *
from utility.rag_evaluation.rag_config import BASELINE_K_SAMPLES
from utility.misc.util import add_zeroes
from langchain.schema.document import Document
import fitz
from io import BytesIO
from PIL import Image

class PDFExtractor:
    """
    Extracts the text and image data from a set of PDF files.

    """
    def __init__(self, 
                 input_folder: Path = DB_SETUP_DATA,
                 projects_folder: Path = APP_PROJECTS) -> None:
        
        self.input_folder = input_folder
        self.projects_folder = projects_folder

        self.pdfs = [file for file in os.listdir(self.input_folder)]
        self.projects_subfolders = []


    def run(self):

        self._create_subfolders()
        self._extract_text()
        self._extract_images()


    def _create_subfolders(self):

        for idx, _ in enumerate(self.pdfs):
            code = add_zeroes(idx) + str(idx)
            subfolder_name = code + "_Project_" + code
            subfolder_path = self.projects_folder / subfolder_name
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"Successfully created -> Folder #{subfolder_name}")

        self.projects_subfolders = [d.name for d in Path(self.projects_folder).iterdir()]

    
    def _extract_text(self) -> None:
        """
        Outputs individual texts to separate files.

        Returns:
            None
        """
        full_text = ""
        full_text_file = DA_TEXT / "full_text_output.txt"

        for idx, f in enumerate(self.pdfs):
            
            # create file index and path
            file_index = add_zeroes(idx) + str(idx)
            path = self.input_folder / f

            # create document by opening pdf with pymupdf
            doc = pymupdf.open(path)

            combined_text = ""
            for page in doc:
                text = page.get_text()
                combined_text += "\n" + text
            
            current_subfolder = self.projects_folder / self.projects_subfolders[idx]
            current_text_file = current_subfolder / f"{file_index}.txt"

            with open(str(current_text_file), "w") as f:
                f.write(combined_text)
                print(f"Successfully extracted -> File #{file_index}.txt")
            
            # exports to data analysis module
            da_text_file = DA_TEXT_IND / f"{file_index}.txt" # files for Data Analysis
            full_text += "\n" + combined_text

            with open(str(da_text_file), "w") as f:
                f.write(combined_text)
                print(f"Successfully extracted -> File #{file_index}.txt")

        # exports to data analysis module
        with open(str(full_text_file), "w") as f:
            f.write(full_text)
            print(f"Successfully extracted -> File 'full_text_output.txt'")


    def _extract_images(self) -> tuple[list[Document], list[str]]:
        """
        Takes a single pdf path and returns each page as a document and the URI.

        Returns:
            tuple[list[Document], list[str]]
        """
        for idx, f in enumerate(self.pdfs):
            file_index = add_zeroes(idx) + str(idx)
            path = self.input_folder / f
            doc = fitz.open(path)

            # with same source
            for page_idx in range(doc.page_count):
                for img_idx, img in enumerate(doc.get_page_images(page_idx, full=True)):
                    
                    # each image on this page is being exported
                    xref = img[0]
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]

                    # opens the image byte data and parses to a PIL image RGB object
                    pil = Image.open(BytesIO(img_bytes)).convert("RGB") 

                    img_id = f"{file_index}:{str(page_idx)}:{str(img_idx)}"
                    img_name = f"{file_index}-{str(page_idx)}-{str(img_idx)}.jpg"
                    img_path = self.projects_folder / self.projects_subfolders[idx] / img_name

                    pil.save(img_path)
                    print(f"Image: {img_id} -> Saved on folder.")