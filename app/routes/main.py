from flask import (Blueprint, request, jsonify, 
                   render_template, redirect, url_for, render_template_string)
from app.routes.components import (create_header, create_footer, 
                                   create_generic_project,
                                   create_rag_interface, get_projects)

main = Blueprint("main", __name__, static_folder="static", template_folder="templates")


# redirect to rag page from a theoretical "home" page
@main.route("/", methods=["GET"])
def home():
    return redirect(url_for("main.rag"))


# rag template with chat box
@main.route("/rag", endpoint="rag", methods=["GET", "POST"])
def rag():

    # POST method logic related to the chat
    if request.method == "POST":
        # fetching user_request from the JS chat
        user_request = request.get_json(force=True)
        # extracting actual message from query
        query = user_request.get("message", "").strip()
        
        # actual RAG querying below
        if query:
            # lazy imports on query
            from services.rag.settings import RagSettings
            from services.rag.query import QueryRag
            # performing query
            settings = RagSettings()
            query_rag = QueryRag(settings)
            answer_with_src = query_rag.query(query)
            # getting answer and project ids from sources
            answer = answer_with_src[0]
            src = answer_with_src[1]
            project_ids = [int(s.split("_")[0]) for s in src]
            cull_duplicates = set(project_ids)
            project_ids = [id for id in cull_duplicates]

            return jsonify({"answer": answer, "project_ids": project_ids})

    # GET logic related to HTML rendering
    # base = create_base()
    header = create_header()
    main = create_rag_interface()
    footer = create_footer()

    return header + main + footer


# routing for refreshing the project thumbnails
@main.route("/rag/projects", methods=["GET"])
def rag_projects_partial():

    # get ids from new sources
    ids = request.args.get("ids", "")
    # generate new data to refresh thumbnail grid
    project_ids = [int(x) for x in ids.split(",") if x.strip()]
    data = get_projects(project_ids)

    return render_template("project_thumbs.html", data=data)


# generic project template
@main.route("/project/<int:id>", endpoint="project_generic")
def project_generic(id):

    header = create_header()
    main = create_generic_project(id)
    footer = create_footer()

    return header + main + footer


# routing for a home page...RAG would be a service to a broader-purpose app
# @main.route("/", endpoint="home")
# def home():

#     header = create_header()
#     main = create_home()
#     footer = create_footer()

#     return  header + main + footer