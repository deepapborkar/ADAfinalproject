'''This file is just for reading in the data and putting it into csv form'''
import os

# NETFLIX DATA
# convert all combined_data_*.txt files in data directory to data.cvs

# get all the data file names
files = []
for file in os.listdir("data"):
    if file.startswith("combined_data"):
        files.append(file)

# read in data to a csv file
f_data = open("data/data.csv","r+")
# clear the data
f_data.truncate(0)

# go through all the files
for file in files:
    file_dir = 'data/' + file
    with open(file_dir, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(',')
        if len(line) == 1:
            movie_id = line[0].split(':')[0]
        else:
            newline = movie_id + ', ' + line[0] + ', ' + line[1] + ', ' + line[2]
            f_data.write(newline)


# TEST DATA
# convert test_data.txt file to test_data.csv 

f_test_data = open("data/test_data.csv","r+")
# clear the data
f_test_data.truncate(0)
# directory of the test data
file_dir = 'data/test_data.txt'
# read in the test data into the test_data.csv
with open(file_dir, 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.split(',')
    if len(line) == 1:
        movie_id = line[0].split(':')[0]
    else:
        newline = movie_id + ', ' + line[0] + ', ' + line[1] + ', ' + line[2]
        f_test_data.write(newline)

# SAMPLE DATA from Netflix Data
# convert sample_data.txt file to sample_data.csv

# directory of the sample data
file_dir = 'data/sample_data.txt'

# keeps track of all of the information in a dictionary
rating_dict = {}

# read in the test data into the sample_data.csv
with open(file_dir, 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.split(',')
    if len(line) == 1:
        movie_id = line[0].split(':')[0]
        rating_dict[movie_id] = []
    else:
        # keeps track of user_id, rating, and date in dictionary with movie_id as key
        rating_dict[movie_id].append([int(line[0]), int(line[1]), line[2]])

#print(rating_dict)

# adds true or false to value depending on if the users have been repeated 
for idx1, val1 in rating_dict.items():
    user_lst1 = [i[0] for i in val1]
    count = [0 for _ in val1]
    for idx2, val2 in rating_dict.items():
        user_lst2 = [i[0] for i in val2]
        for i, u in enumerate(user_lst1):
            if u in user_lst2:
                count[i] += 1
    for i, c in enumerate(count):
        if c > 1:
            val1[i].append(True)
        else:
            val1[i].append(False)

#print(rating_dict)

final_rating_dict = {}

# keep a dictionary of only the users to include (that have multiple ratings)
for idx, val in rating_dict.items():
    final_rating_dict[idx] = []
    for i, v in enumerate(val):
        if v[3]:
            final_rating_dict[idx].append([v[0], v[1], v[2]])

#print(final_rating_dict)

users_lst = []

# get a list of all users
for idx, val in final_rating_dict.items():
    for v in val:
        users_lst.append(v[0])
# get a list of the unique users
unique_users = list(set(users_lst))

#print(unique_users)

# update the users to new idx values starting with index 1
for idx, val in final_rating_dict.items():
    for i, v in enumerate(val):
        new_user_id = unique_users.index(v[0]) + 1
        final_rating_dict[idx][i][0] = new_user_id

#print(final_rating_dict)

f_samp_data = open("data/sample_data.csv","r+")
# clear the data
f_samp_data.truncate(0)

# write the data back to the csv file
for idx, val in final_rating_dict.items():
    for i, v in enumerate(val):
        newline = idx + ', ' + str(v[0]) + ', ' + str(v[1]) + ', ' + v[2]
        f_samp_data.write(newline)