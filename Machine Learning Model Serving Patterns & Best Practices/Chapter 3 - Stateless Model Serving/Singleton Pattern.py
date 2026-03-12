#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 3: Stateless Model Serving
"""

class DBConnection(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            print('Creating database connection...')
            cls.instance = super(DBConnection, cls).__new__(cls)
            return cls.instance
        else:
            print('Database connection has already been established!')
            return cls.instance

if __name__ == "__main__":
    connection1 = DBConnection()
    connection2 = DBConnection()
    connection3 = DBConnection()