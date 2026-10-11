from flask import render_template
from sqlalchemy import select
from app.database import (ProjectReader, ProjectImageReader, ProjectList, Project)
from app.extensions import db

# render header
def create_header():
    return render_template('header.html')


# render footer
def create_footer():
    return render_template('footer.html')


# RAG results section
def create_rag_interface(project_ids: list | None = None):
    if project_ids is not None:
        data = get_projects(project_ids)
        return render_template('rag.html', data=data)
    else:
        projects = ProjectList(db.session)
        data =  get_projects(projects.ids)
        return render_template('rag.html', data=data)


# get projects information
def get_projects(project_ids):
    hero_images = []
    project_names = []
    project_categories = []
    for project_id in project_ids:
        reader = ProjectImageReader(db.session, project_id)
        hero = reader.image_paths[0]
        hero_images.append(hero)

        # new logic for project names
        stmt = select(Project).filter_by(id=project_id)
        project = db.session.scalars(stmt).first()
        project_names.append(project.name)
        project_categories.append(project.category_id)

    print(f"Creating projects...")
    print(f"Image path: {hero_images}")

    data = zip(hero_images, project_ids, project_names, project_categories)
    return data


# an individual project page
def create_generic_project(id):
        project_reader = ProjectReader(db.session, id)
        image_reader = ProjectImageReader(db.session, id)
        name = project_reader.name
        summary = project_reader.summary
        images = image_reader.image_paths
        return render_template('project_generic.html', name=name, 
                               summary=summary, images=images)