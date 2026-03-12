#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 6: Batch Model Serving
"""
from crontab import CronTab

cron = CronTab(user = True)
job = cron.new(command = "echo Hello World! | wall")
# job.minute.every(1)
cron.remove_all()
cron.write()