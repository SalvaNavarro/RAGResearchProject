from app import create_app
from app.extensions import db
from utility.db_setup.seed import Seed

def main():
    # resetting the entire database on every run
    app = create_app()
    extract_pdf = True # optionally set to false after first run
    with app.app_context():
        db.drop_all()
        db.create_all()
        seed = Seed(extract_pdf)
        seed.populate_db(db)

if __name__ == '__main__':
    main()