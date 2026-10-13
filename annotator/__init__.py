"""Local browser tool for reviewing and editing the welding annotations.

    data.py     finds images, human masks, COCO JSONs and the folder being
                annotated; builds instances; saves RGB masks + COCO JSON
    server.py   a small local web server (standard library only)
    static/     the browser app (index.html, app.js, style.css)

Start it with annotate.py in the repository root.
"""
