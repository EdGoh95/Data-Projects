#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 4: Data Acquisition, Data Quality, and Noise
"""
import logging
# Create a logging file with the format of the log messages
logging.basicConfig(filename = 'Information Quality (Gerrit).log',
                    format = '%(asctime)s; %(name)s;%(levelname)s;%(message)s', level = logging.DEBUG)
from pygerrit2 import GerritRestAPI
from atlassian import Jira
from github import Github
from difflib import unified_diff

#%% Data Extraction From Software Engineering Tools
#### Gerrit - For Code Reviews
gerrit_logger = logging.getLogger("Gerrit Data Export Pipeline")
gerrit_logger.info("Configuration started")

gerrit_url = "https://gerrit.onap.org/r"
auth = None # Gerrit is a public OSS repository
gerrit_API = GerritRestAPI(url = gerrit_url, auth = auth)
gerrit_logger.info("REST API setup complete")

gerrit_logger.info("Connecting to Gerrit server and accessing changes")
start = 0
try:
    changes = gerrit_API.get(
        "/changes/?q=status:merged&o=ALL_FILES&o=ALL_REVISIONS&o=DETAILED_LABELS&start={}".format(start),
        headers = {'Content-Type': 'application/json'})
except Exception as e:
    gerrit_logger.error("ENTITY ACCESS - Error retrieving changes: {}".format(e))
gerrit_logger.info("Changes retrieved")
logging.shutdown()

#### JIRA - Issue & Task Management Tool Developed By Atlassian
# JIRA_instance = Jira(url = "https://infocepts-teams.atlassian.net", # URL of the JIRA server
#                     username = 'edwin.goh@infocepts.com', password = <YOUR_TOKEN>, cloud = True)

#%% Data Extraction From Product Databases - GitHub & Git
git_pat = "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"
git_instance = Github(git_pat, per_page = 100)

#### From My Repository
my_repo = git_instance.get_repo("EdGoh95/Data-Projects")
my_commits = my_repo.get_commits()
print('Repository name:', my_repo.full_name)
print('Number of commits:', my_commits.totalCount)
print('Commit message:', my_commits[3].commit.message)
print('Number of files pushed in commit:', my_commits[3].files.totalCount)

my_pushed_file = my_commits[3].files[0]
my_pushed_file_content = my_repo.get_contents(my_pushed_file.filename, ref = my_commits[3].sha)
my_pushed_file_content_decoded = my_pushed_file_content.decoded_content.decode('utf-8')

#### From The Book's Repository
repo = git_instance.get_repo("miroslawstaron/machine_learning_best_practices")
commits = repo.get_commits()
print('\nRepository name:', repo.full_name)
print('Number of commits:', commits.totalCount)
print('Commit message:', commits[0].commit.message)
print('Number of files pushed in commit:', commits[0].files.totalCount)

commit = commits[4].files[0]
following_commit = commits[5].files[0]
commit_file_content = repo.get_contents(commit.filename, ref = commits[4].sha)
following_commit_file_content = repo.get_contents(following_commit.filename, ref = commits[5].sha)
commit_file_content_decoded = commit_file_content.decoded_content
following_commit_file_content_decoded = following_commit_file_content.decoded_content

addition = []
deletion = []
for line in unified_diff(str(following_commit_file_content_decoded), str(commit_file_content_decoded),
                          fromfile = following_commit.filename, tofile = commit.filename):
    if line[0] == '+':
        addition.append(line[1])
    if line[0] == '-':
        deletion.append(line[1])
print('\nModifed lines:', ''.join(deletion), ''.join(addition), sep = '\n')