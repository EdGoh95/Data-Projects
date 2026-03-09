"""
This script reads the Disk Utilization statistics for each server in the 'Daily Server Monitoring Alerts'
folder and automatically populates the Disk Utilization Monitoring.xlsx Excel file
"""
import sys
import os
import time
import pandas as pd
import xlrd3
import openpyxl
from datetime import date
today_date = date.today().strftime('%a, %d-%b-%y')

def matching_server(server):
    '''
    Determine the row number in the Excel spreadsheet where the server name (first column) resides
    '''
    for row in range(environment_sheet.nrows):
        if environment_sheet.cell_value(row, 0) == server:
            return row
    return "Server cannot be found!"

def matching_mount(start, end, mount_points):
    '''
    Determine the row number where a particular mount point is located
    The output list follows the order given by mount_dict_sorted
    '''
    rows = []
    for mount in mount_points:
        for row in range(start + 1, end):
            if environment_sheet.cell_value(row, 0) == mount:
                rows.append(row)
    return rows

def new_section(df, key):
    '''
    Determine a new section based on the key, which also acts as a delimiter
    '''
    return df[df[0].str[0] == key].index

# Open the Excel workbook and store the sheets in a list (May need to exclude the last sheet)
Disk_Utilization_Monitoring = xlrd3.open_workbook('Disk Utilization Monitoring.xlsx')
environments = Disk_Utilization_Monitoring.sheet_names()
# Exclude the "Legend" spreadsheet (for reference only)
environments.pop()

# Looping through all the spreadsheets where each sheet represents each environment
for environment in environments:
    start = time.perf_counter()
    print("\nEnvironment: " + environment)
    environment_sheet = Disk_Utilization_Monitoring.sheet_by_name(environment)
    # Determine the column number containing today's date
    for col in range(environment_sheet.ncols):
        if environment_sheet.cell_value(0, col) == today_date:
            break

    environment_folder = 'Daily Server Monitoring Alerts/{}/{}/'.format(today_date, environment)
    # Initialize dictionaries whose keys are the server names (in upper case)
    server_dict = {}
    mount_dict = {}
    usage_dict = {}
    for file in os.listdir(environment_folder):
        print('Processing {}:'.format(file.split('_')[0].upper()))
        server_dict[file.split('_')[0].upper()] = matching_server(file.split('_')[0].upper())
        server_monitoring_report_df = pd.read_csv(environment_folder + file, header = None,
                                                  on_bad_lines = 'skip')
        new_section_line_numbers = new_section(server_monitoring_report_df, '*')
        # Exclude the header line of the "df -kh" command and store the table in a new dataframe
        mount_disk_utilization = server_monitoring_report_df.iloc[
            new_section_line_numbers[0]+2:new_section_line_numbers[1]].reset_index(drop = True)

        mounts = []
        usage = []
        print('Extracting mount points and usage...')
        for index, line in mount_disk_utilization.iterrows():
            # The mount points are in the last row of the Disk Utilization table
            mounts.append(line.str.split()[0][-1])
            # The mount usages are in the second last row of the Disk Utilization table
            usage.append(line.str.split()[0][-2])
        mount_dict[file.split('_')[0].upper()] = mounts
        usage_dict[file.split('_')[0].upper()] = usage

    # Sort the dictionaries according to the order specified in the Excel spreadsheets
    server_list_sorted = sorted(server_dict, key = server_dict.get)
    server_dict_sorted = {}
    mount_dict_sorted = {}
    usage_dict_sorted = {}
    for server in server_list_sorted:
        server_dict_sorted[server] = server_dict[server]
        mount_dict_sorted[server] = mount_dict[server]
        usage_dict_sorted[server] = usage_dict[server]

    matching_rows_dict = {}
    for index, (server_name, row_number) in enumerate(server_dict_sorted.items()):
        if index < len(server_dict_sorted) - 1:
            matching_rows_dict[server_name] = matching_mount(
                list(server_dict_sorted.values())[index], list(server_dict_sorted.values())[index + 1],
                mount_dict_sorted[server_name])
        else:
            matching_rows_dict[server_name] = matching_mount(
                list(server_dict_sorted.values())[index], environment_sheet.nrows,
                mount_dict_sorted[server_name])

    mount_usage_dict = {}
    '''
    Extract the row numbers of the respective mount points and drop any mount points that are not
    present in the Excel file
    '''
    for server, mount_usage_list in usage_dict_sorted.items():
        row_ids = matching_rows_dict[server]
        mount_usage_dict[server] = list(zip(row_ids, mount_usage_list[0:len(row_ids)]))

    usage_row_ids = []
    # Combine the tuples in the mount_usage_dict into a single list for easy extraction
    for server, usage_tuple in mount_usage_dict.items():
        for usage_rowid in usage_tuple:
            usage_row_ids.append(usage_rowid)

    Disk_Utilization_Monitoring_edit = openpyxl.load_workbook('Disk Utilization Monitoring.xlsx')
    environment_sheet_edit = Disk_Utilization_Monitoring_edit[environment]
    print('Populating Disk Utilization For {}...'.format(environment))
    for rowid, mount_usage in usage_row_ids:
        environment_sheet_edit.cell(row = rowid + 1, column = col + 1).value = mount_usage
    Disk_Utilization_Monitoring_edit.save('Disk Utilization Monitoring.xlsx')
    stop = time.perf_counter()
    print('Processing Time: {:.2f}s'.format(stop - start))

Disk_Utilization_Monitoring.release_resources()
Disk_Utilization_Monitoring_edit.close()
del Disk_Utilization_Monitoring, Disk_Utilization_Monitoring_edit

# Restart the python console to close and release the Excel file from the program
os.execv(sys.executable, ['python'] + sys.argv)
