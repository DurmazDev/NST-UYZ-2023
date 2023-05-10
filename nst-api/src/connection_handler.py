import json
import logging
import requests
from decouple import config

class ConnectionHandler:
    config.search_path = "./configs/"
    def __init__(self, base_url=config('EVALUATION_SERVER_URL'), username=None, password=None):
        self.base_url = base_url
        self.auth_token = None
        self.classes = None
        self.frames = None

        # Define URLs
        self.url_login = self.base_url + "auth/"
        self.url_frames = self.base_url + "frames/"
        self.url_prediction = self.base_url + "prediction/"

        if username and password:
            self.login(username, password)

    def login(self, username, password):
        payload = {
            'username': username,
            'password': password
        }
        files = []
        response = requests.request("POST", self.url_login, data=payload, files=files, timeout=3)
        response_json = json.loads(response.text)
        if response.status_code == 200:
            self.auth_token = response_json['token']
            logging.info("Login Successfully Completed : {}".format(payload))
        else:
            logging.info("Login Failed : {}".format(response.text))

    def get_frames(self):
        payload = {}
        headers = {'Authorization': 'Token {}'.format(self.auth_token)}

        response = requests.request("GET", self.url_frames, headers=headers, data=payload)
        self.frames = json.loads(response.text)

        if response.status_code == 200:
            logging.info("Successful : get_frames : {}".format(self.frames))
        else:
            logging.info("Failed : get_frames : {}".format(response.text))
        return self.frames

    def send_prediction(self, prediction):
        payload = json.dumps(prediction.create_payload(self.base_url))
        print(payload)
        files = []
        headers = {
            'Authorization': 'Token {}'.format(self.auth_token),
            'Content-Type': 'application/json',
        }
        response = requests.request("POST", self.url_prediction, headers=headers, data=payload, files=files)
        if response.status_code == 201:
            logging.info("Prediction send successfully. \n{}".format(payload))
        else:
            logging.info("Prediction send failed. \n{}".format(response.text))
        return response
