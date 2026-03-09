#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 16: Integrating ML Systems in Ecosystems
"""
import requests

flask_web_service_url = 'http://localhost:5000/success'
ML_web_app_url = 'http://localhost:5001/predict'

def upload_file_and_get_metrics(filepath):
    try:
        with open(filepath, 'rb') as file:
            # Create a dictionary to hold the file data
            files = {'file': (file.name, file)}

            response = requests.post(flask_web_service_url, files = files)
            response.raise_for_status()

            result = response.json()
            LOC = result.get('Lines of Code')
            MCC = result.get('McCabe Complexity')[0][-1]
            if LOC is not None and MCC is not None:
                print('Lines of Code: {:d}, McCabe Complexity: {}'.format(LOC, MCC))
                return LOC, MCC
            else:
                print('Lines of Code and McCabe Complexity cannot be found in the JSON response.')
    except Exception as e:
        print('Error:', e)

def send_metrics_for_predictions(LOC, MCC):
    try:
        prediction_url = '{}/{}/{}'.format(ML_web_app_url, LOC, MCC)

        prediction_response = requests.get(prediction_url)
        prediction_response.raise_for_status()

        prediction = prediction_response.json().get('Defect')

        print("Prediction:", prediction)

    except Exception as e:
        print('Error:', e)

if __name__ == '__main__':
    LOC, MCC = upload_file_and_get_metrics('nx_ip_checksum_compute.c')
    send_metrics_for_predictions(LOC, MCC)