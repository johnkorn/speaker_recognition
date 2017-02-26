import os
from speaker_verifier import SpeakerVerifier
from codecs import open
import time
from flask import Flask, render_template, request
import re
app = Flask(__name__)

print "Preparing speaker verifier"
start_time = time.time()
verifier = SpeakerVerifier()
print "The model is ready"
print time.time() - start_time, "seconds"
test_file_name = 'tmp_test.wav' # PATH for saving wav files that are passed during verification stage
train_folder_name = verifier.get_train_path()


def is_int(x):
	match = re.search("\D", x)
	if not match:
		return True
	else:
		return False

@app.route("/speaker-verifier-demo", methods=["POST", "GET"])
def index_page(prediction="", users=""):
	users = verifier.get_users()
	result = ''
	if request.method == "POST":
		if request.form['bsubmit'] == 'Reset': 
			users = verifier.reset_classifier()

		elif request.form['bsubmit'] == 'Upload': # DO TESTING			
			f = request.files['file_for_test']
			if f.filename == '':
				print('No selected file')
			if f.filename.find('wav')==-1 and f.filename.find('flac')==-1:
				print('Selected file is not in audio format (WAV or FLAC)!')
			f.save(test_file_name)
			result = verifier.verify(test_file_name)

		elif request.form['bsubmit'] == 'Upload files': # DO TRAINING
			uploaded_files = request.files.getlist("files_for_training")
			userid = request.form["userid"]

			filenames = []
			for file in uploaded_files:
				# Check if the file is one of the allowed types/extensions
				if file and (file.filename.find('wav')>0 or file.filename.find('flac')>0):
					filename = file.filename
					# Move the file form the temporal folder to the upload folder 
					full_path = os.path.join(train_folder_name, filename)
					file.save(full_path)
					# Save the filename into a list, we'll use it later
					filenames.append(full_path)					
				else:
					print('Selected file "%s" is not in WAV or FLAC format!' % file.filename)

			if (is_int(userid)) and (int(userid) in users.keys()):
				verifier.train_existing_user(filenames, int(userid))
			else:
				verifier.train_new_user(filenames, userid)
				users = verifier.get_users()
			

	return render_template('hello.html', prediction=result, users=users)


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=6006, debug=False)
