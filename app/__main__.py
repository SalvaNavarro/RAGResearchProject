import webbrowser # for development only, it automatically opens the browser
import os
import argparse
from app import create_app

def main(conf: str | None = None) -> None:
    """
    Executes the app.

    """

    app = create_app()

    if conf == "docker":
        host = os.environ.get('SERVER_HOST', '0.0.0.0')
        try:
            port = int(os.environ.get('PORT', '8080'))
        except ValueError:
            port = 8080

        app.run(host, port, debug=False, use_reloader=False)

    else:
        host = os.environ.get('SERVER_HOST', '127.0.0.1')
        try:
            port = int(os.environ.get('SERVER_PORT', '5000'))
        except ValueError:
            port = 5000

        webbrowser.open(f"http://{host}:{port}/rag") # this line for development only
        app.run(host, port, debug=False, use_reloader=False)


if __name__ == '__main__':

    # arguments for docker export
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", help="Toggles docker settings mode.", 
                        default=None)
    args = parser.parse_args()

    main(args.docker)