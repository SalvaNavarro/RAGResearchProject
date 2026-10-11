# main file to populate the database
from utility.db_setup.pdf_extractor import PDFExtractor
from app import create_app
from sqlalchemy import text
import re
import os
from config import *
from utility.misc.util import add_zeroes

class Seed:

    def __init__(self, extract_pdf: bool=False, mode: str="production"):
        
        if extract_pdf:
            pe = PDFExtractor()
            pe.run()

        if mode == "production":
            self.folder = APP_PROJECTS
            self.projects = [d.name for d in APP_PROJECTS.iterdir() if d.is_dir()]
        elif mode == "test":
            self.folder = TEST_APP_PROJECTS
            self.projects = [d.name for d in TEST_APP_PROJECTS.iterdir() if d.is_dir()]
            
        self.categories = ["General"] # optionally assign categories to projects
        self.mode = mode


    def populate_db(self, db):
        # dev logic, will drop and re-create project table every time + populate
        app = create_app(self.mode)
        with app.app_context():

            for category in self.categories:
                category_id = self.categories.index(category)
                category_name = category

                # populate category table
                stmt = text(
                    "INSERT INTO category (id, name) VALUES (:id, :name);"
                )
                db.session.execute(stmt, {
                    "id": category_id,
                    "name": category_name
                })
                db.session.commit()


            # inserting projects
            for project in self.projects:

                project_id = self.projects.index(project)
                project_name = self._create_project_name(project)
                project_path_flask = "projects/" + project + "/"
                project_path = str(self._create_project_path(project))
                project_summary = self._get_project_summary(project)

                # populate project table
                stmt = text(
                    "INSERT INTO project (id, name, path, summary) VALUES (:id, :name, :path, :summary);"
                )
                db.session.execute(stmt, {
                    "id": project_id,
                    "name": project_name,
                    "path": project_path_flask,
                    "summary": project_summary
                })
                db.session.commit()

                # populate project images
                project_images = self._get_image_names(project_path)
                print(project_images)

                for image in project_images:
                    # image_id = project_images.index(image)
                    stmt_images = text(
                        "INSERT INTO project_images (project_id, file_name) VALUES (:project_id, :file_name);"
                    )
                    db.session.execute(stmt_images, {
                        # "id": image_id,
                        "project_id": project_id,
                        "file_name": str(project_id) + '_' + image,
                    })
                    db.session.commit()


    # project name string creation
    def _create_project_name(self, folder_name):
        pattern = r'(^[^_]*_|_)'
        replacement = ' '
        return re.sub(pattern, replacement, folder_name).lstrip()


    # project path string creation
    def _create_project_path(self, folder_name):
        path = self.folder / folder_name
        return path


    def _get_image_names(self, path):
        image_names = []
        for entry in os.listdir(path):
            # workaround to get the text files from single source of truth
            if os.path.isfile(os.path.join(path, entry)) and not entry.lower().endswith('.txt'): 
                image_names.append(entry)
        return image_names
    

    def _get_project_summary(self, folder_name):
        summary_dir = self.folder / self._create_project_path(folder_name)
        for entry in os.listdir(summary_dir):
            if entry.lower().endswith('.txt'): 
                summary_path = summary_dir / entry
                try:
                    with open (summary_path, 'r', 
                               encoding="utf-8", errors="replace") as file:
                        summary = file.read()
                        return summary
                except FileNotFoundError:
                    print("File not found")


    def _assign_category(self, project_name, categories):
        ...