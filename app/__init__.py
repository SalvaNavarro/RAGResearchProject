from flask import Flask
from app.extensions import db
from app.routes.main import main
from config import ProductionConfig, TestingConfig


def create_app(mode="production"):
    """
    Creates the app, factory pattern

    Args:
        config: Configuration object.
    
    Returns:
        Flask: The flask app.
    """

    # initialise app
    app = Flask(__name__)

    # apply configurations from config object depending on mode
    if mode == "test":
        app.config.from_object(TestingConfig())  
    else:
        app.config.from_object(ProductionConfig())

    # intialise app db and register blueprint
    db.init_app(app)

    # register blueprint
    app.register_blueprint(main, url_prefix="")

    return app