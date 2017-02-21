from speaker_verifier import SpeakerVerifier
from codecs import open
import time
from flask import Flask, render_template, request
app = Flask(__name__)

print "Preparing speaker verifier"
start_time = time.time()
verifier = SpeakerVerifier()
print "The model is ready"
print time.time() - start_time, "seconds"

@app.route("/speaker-verifier-demo", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("speaker_demo_logs.txt", "a", "utf-8")
	print text
	print >> logfile, "<response>"
	print >> logfile, text
        prediction_message = classifier.get_prediction_message(text)
        print prediction_message
	print >> logfile, prediction_message
	print >> logfile, "</response>"
	logfile.close()
	
    return render_template('hello.html', text=text, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=False)
