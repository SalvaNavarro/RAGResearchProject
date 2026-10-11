# ORM DB classes
from app.extensions import db
from sqlalchemy import select


class Category(db.Model):
    """Table category."""

    __tablename__ = 'category'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(150), nullable=False, unique=True)

    projects = db.relationship(
        "Project",
        back_populates="category",
        cascade="all, delete-orphan"
    )


class Project(db.Model):
    """Table project."""

    __tablename__ = 'project'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(150), nullable=False, unique=True)
    category_id = db.Column(
        db.Integer,
        db.ForeignKey('category.id'),
        nullable=True
    )
    confidential = db.Column(db.Boolean, nullable=True, default=False)
    path = db.Column(db.String(150), nullable=False, unique=True)
    summary = db.Column(db.Text) # text description

    # relationship pointers
    images = db.relationship(
        "ProjectImage",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    category = db.relationship(
        "Category",
        back_populates="projects",
        lazy="joined"
    )


class ProjectImage(db.Model):
    """Table project image."""

    __tablename__ = 'project_images'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = db.Column(db.Integer, 
                           db.ForeignKey("project.id", ondelete="CASCADE"), 
                           nullable=False)
    file_name = db.Column(db.String(150), nullable=False, unique=True)

    # relationship pointer to parent class
    project = db.relationship("Project", back_populates="images")


class User(db.Model):

    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(150), nullable=False, unique=True)

# CRUD operations
class ProjectList:
    """Retrieval of project ids"""

    def __init__(self, session):
        self.session = session
        self.ids = self._get_project_id_list()

    # returns the project list
    def _get_project_id_list(self):
        stmt = select(Project.id)
        project_ids = self.session.scalars(stmt).all()
        return project_ids


class ProjectReader:
    """Retrieval of project information."""

    def __init__(self, session, project_id: int):
        self.session = session
        self.id = project_id
        self.name = self._get_project_name()
        self.path = self._get_project_path()
        self.summary = self._get_project_summary()

    # returns the file names as list str
    def _get_project_name(self):
        project_name_row = self.session.query(Project.name).filter(Project.id == self.id).first()
        project_name = project_name_row[0]
        return project_name
    
    # returns the file names as list str
    def _get_project_path(self):
        project_path_row = self.session.query(Project.path).filter(Project.id == self.id).first()
        project_path = project_path_row[0]
        return project_path
    
    # returns the project summary
    def _get_project_summary(self):
        project_summary_row = self.session.query(Project.summary).filter(Project.id == self.id).first()
        project_summary = project_summary_row[0]
        return project_summary


class ProjectImageReader:
    """Retrieval of project image information."""

    def __init__(self, session, project_id: int):
        self.session = session
        self.project = ProjectReader(session, project_id)
        self.image_names = self._get_image_names()
        self.image_paths = self._get_image_paths()

    # querying images from DB
    def _get_image_names(self):
        stmt = select(ProjectImage).filter_by(project_id=self.project.id)
        project_img_models = self.session.scalars(stmt).all()
        img_names = []
        for model in project_img_models:
            img_names.append(model.file_name)
        return img_names

    def _get_image_paths(self):
        project_path = self.project.path
        image_paths = []
        for name in self.image_names:
            # removing prefixed int that was added to meet unique constraint requirement
            short_name = name.find("_")
            new_name = name[short_name + 1:]
            path = project_path + new_name
            image_paths.append(path)
        return image_paths