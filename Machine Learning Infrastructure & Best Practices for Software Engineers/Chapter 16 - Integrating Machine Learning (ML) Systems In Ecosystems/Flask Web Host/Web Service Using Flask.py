#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 16: Integrating ML Systems in Ecosystems
"""
from flask import *
from radon.complexity import cc_visit
from radon.cli.harvest import CCHarvester

web_app = Flask(__name__)

# Initializing a dictionary to store the metrics (lines of code and McCabe complexity) for the submitted file
code_metrics = {}

def calculate_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Count the number of lines of code
    lines = len(content.splitlines())
    McCabe_complexity = cc_visit(content)
    code_metrics[file_path] = {'Lines of Code': lines, 'McCabe Complexity': McCabe_complexity}

@web_app.route('/')
def main():
    return render_template('index.html')

@web_app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        file = request.files['file']
        # Save the file to the web server
        file_path = file.filename
        file.save(file_path)
        # Calculate the metrics for the file
        calculate_metrics(file_path)
        return code_metrics[file_path]

@web_app.route('/metrics', methods = ['GET'])
def get_metrics():
    if request.method == 'GET':
        return code_metrics

if __name__ == '__main__':
    web_app.run(host = '0.0.0.0', debug = True)