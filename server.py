import os
from flask import Flask, render_template, request
from flask_restful import Resource, Api, abort, reqparse
from speaker_verifier import SpeakerVerifier
import re
import random

app = Flask(__name__)
api = Api(app)

verifier = SpeakerVerifier()
full_path = os.path.realpath(__file__)
test_file_name = 'tmp_test.wav'
train_folder_name = verifier.get_train_path()


class UsersResource(Resource):
    def get(self):
        users = verifier.get_users()
        return [{
            'key': key,
            'value': value
        } for key, value in users.items()]


class ResetResource(Resource):
    def post(self):
        verifier.reset_classifier()
        return {}


class VerifyResource(Resource):
    def post(self):
        f = request.files['file_for_test']
        if f.filename == '':
            abort(400, message='No selected file')
        if f.filename.find('wav') == -1 and f.filename.find('flac') == -1:
            abort(400, message='Selected file is not in audio format (WAV or FLAC)!')
        f.save(test_file_name)
        result = verifier.verify(test_file_name)
        return {'verify': result}


class TrainResource(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('userid', type=int)
        args = parser.parse_args(strict=True)
        userid = args["userid"]
        uploaded_files = request.files.getlist("files_for_training")
        filenames = []
        for file in uploaded_files:
            if file and (file.filename.find('wav') > 0 or file.filename.find('flac') > 0):
                filename = file.filename
                full_path = os.path.join(train_folder_name, filename)
                file.save(full_path)
                filenames.append(full_path)
            else:
                abort(400, message="Selected file {} is not in WAV or FLAC format!".format(file.filename))

        if (userid) and (userid in db_users.keys()):
            verifier.train_existing_user(filenames, userid)
        else:
            verifier.train_new_user(filenames, userid)
            users = verifier.get_users()
        return [{
            'key': key,
            'value': value
        } for key, value in users.items()]


api.add_resource(ResetResource, '/api/reset')
api.add_resource(VerifyResource, '/api/verify')
api.add_resource(TrainResource, '/api/train')
api.add_resource(UsersResource, '/api/users')


def is_int(x):
    match = re.search("\D", x)
    if not match:
        return True
    else:
        return False

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=True)
