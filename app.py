from flask import Flask, render_template, request
import numpy as np
from simplex import simplex

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Récupération des données du formulaire
            A = eval(request.form["A"])
            b = eval(request.form["b"])
            c = eval(request.form["c"])

            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
            c = np.array(c, dtype=float)

            x, z = simplex(c, A, b)
            if x is None:
                result = {"error": "Problème non borné."}
            else:
                result = {"x": x.tolist(), "z": -z}
        except Exception as e:
            result = {"error": str(e)}
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True ,use_reloader=False)
