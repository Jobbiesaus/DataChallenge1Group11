from flask import Flask, flash, render_template, request, session, redirect

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    """
    :return: The home web page.
    """
    if request.method == 'POST': 
        if request.form["button"] == "Predict":
            data = request.form["x"]
            return render_template("test.html", data = data)
    else:
        return render_template("WebsiteEyeDoctor.html")

if __name__ == "__main__":
    app.run(debug=True)