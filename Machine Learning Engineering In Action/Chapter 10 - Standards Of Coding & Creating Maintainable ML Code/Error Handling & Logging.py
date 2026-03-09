#!/usr/bin/env python3
import logging
import inspect
import numpy as np
import random
from datetime import datetime
from importlib import reload

reload(logging)
log_filename = 'Logs/{}.log'.format(datetime.now().strftime('%d-%b-%y-%H%Mhrs'))
logging.basicConfig(filename = log_filename, format = 'ERROR: %(message)s', level = logging.ERROR)

class CalculationError(ValueError):
    def __init__(self, message, pre, post, *args):
        self.message = message
        self.pre = pre
        self.post = post
        super(CalculationError, self).__init__(message, pre, post, *args)

def divide_values(base, collection):
    function_name = inspect.currentframe().f_code.co_name
    output = []
    for i in collection:
        try:
            output.append(base/i)
        except ZeroDivisionError as e:
            logging.error("{}: {} - Couldn't divide {} by {} due to {} in {}".format(
                datetime.now().strftime('%d/%m/%Y %H:%M:%S'), type(e), base, i, e, function_name))
        except TypeError as e:
            logging.error("{}: {} - Couldn't process the base value '{}' ({}) in {}".format(
                datetime.now().strftime('%d/%m/%Y %H:%M:%S'), type(e), base, e, function_name))
            raise e
    input_length = len(collection)
    output_length = len(output)
    if input_length != output_length:
        error_message = ("The return size of the collection doesn't match that was passed in "
                         "the collection.")
        e = CalculationError(error_message, input_length, output_length)
        logging.error('{}: {} - Input: {}, Output: {} in {}'.format(
            datetime.now().strftime('%d/%m/%Y %H:%M:%S'), e.message, e.pre, e.post, function_name))
        raise e
    return output

def read_log(log_filename):
    try:
        with open(log_filename) as log:
            print(log.read())
    except FileNotFoundError:
        print('The log file seems to be empty. Please check input filepath.')