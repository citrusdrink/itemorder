import csv
from operator import add
from numpy import mean
from scipy.stats import ttest_rel
from functools import reduce
import itertools
import csv
import json
import math

base_output_folder = "base2015/"
transitions_output_folder = "order2015/"
interesting_bsi_file = "interestingbsisfinal.csv"
original_data_folder = "2015updates/"

base_models = {} #from bsi to dictionary of interesting things
transition_models = {}

with open(interesting_bsi_file,newline='') as csvfile:
	datafile = csv.reader(csvfile)
	first_row = next(datafile)
	interesting_bsis = list(first_row)

interesting_things = ['guesses', 'learns', 'loglike', 'prior', 'slips']

for bsi in interesting_bsis:
	base_models[bsi] = {}
	transition_models[bsi] = {}
	for thing in interesting_things:
		base_models[bsi][thing] = []
		with open(base_output_folder + 'b' + bsi + thing + ".csv") as csvfile:
			datafile = csv.reader(csvfile)
			for row in datafile:
				for elem in row:
					base_models[bsi][thing].append(elem)
		transition_models[bsi][thing] = []
		with open(transitions_output_folder + 'o' + bsi + thing + ".csv") as csvfile:
			datafile = csv.reader(csvfile)
			for row in datafile:
				for elem in row:
					transition_models[bsi][thing].append(elem)
	with open(original_data_folder + bsi + "reverse_orderings2015.json") as f:
		data = json.load(f)
		transition_models[bsi]['reverse_orderings'] = data
		transition_models[bsi]['ordering_to_resource'] = {v: k for k, v in data.items()}

template_count_to_total_resources = {2: 6, 3: 12, 4: 20, 5:30, 6:42}
total_resources_to_template_count = {6: 2, 12: 3, 20: 4, 30: 5, 42:6}
relevant_resources = {} #key is bsi
for bsi in interesting_bsis:
	learncount = len(transition_models[bsi]['learns'])
	temp_count = total_resources_to_template_count[learncount]
	relevant_resources[bsi] = range(2, learncount-temp_count+2)

base_output_folder2 = "basecross2015/"
transitions_output_folder2 = "ordercross2015/"
interesting_bsi_file = "interestingbsisfinal.csv"

with open(interesting_bsi_file,newline='') as csvfile:
	datafile = csv.reader(csvfile)
	first_row = next(datafile)
	interesting_bsis = list(first_row)

bsi_dict = {}

cv_partitions = ["c1", "c2", "c3", "c4", "c5"]

training_resource_sets = {}
test_resource_sets = {}
for bsi in interesting_bsis:
    for cv in cv_partitions:
        with open("ordercross2015/o"+bsi+cv+"trainingresources.csv",newline='') as csvfile:
            datafile = csv.reader(csvfile)
            first_row = next(datafile)
            training_set = set()
            for rid in first_row:
                training_set.add(rid)
            training_resource_sets[bsi+cv] = training_set
        with open("ordercross2015/o"+bsi+cv+"testresources.csv",newline='') as csvfile:
            datafile = csv.reader(csvfile)
            first_row = next(datafile)
            test_set = set()
            for rid in first_row:
                test_set.add(rid)
            test_resource_sets[bsi+cv] = test_set

for bsi in interesting_bsis:
	bsi_dict[bsi] = {}
	current_bsi_dict = bsi_dict[bsi]
	for cv in cv_partitions:
		with open(base_output_folder2 + 'b' + bsi + cv + "data.csv") as csvfile:
			datafile = csv.reader(csvfile)
			responses = []
			for row in datafile:
				responses.append(row)
			current_bsi_dict[cv+"data"] = responses

		with open(base_output_folder2 + 'b' + bsi + cv + "predictions.csv") as csvfile:
			datafile = csv.reader(csvfile)
			basepredictions = []
			for row in datafile:
				basepredictions.append(row)
			current_bsi_dict[cv+"basepredictions"] = basepredictions

		with open(transitions_output_folder2 + 'o' + bsi + cv + "predictions.csv") as csvfile:
			datafile = csv.reader(csvfile)
			orderpredictions = []
			for row in datafile:
				orderpredictions.append(row)
			current_bsi_dict[cv+"orderpredictions"] = orderpredictions

		with open(transitions_output_folder2 + 'o' + bsi + cv + "testresources.csv") as csvfile:
			datafile = csv.reader(csvfile)
			row = next(datafile)
			current_bsi_dict[cv+"resources"] = row

error_rates = {}
for bsi in bsi_dict:
	current_bsi_dict = bsi_dict[bsi]
	error_rates[bsi] = {} #keys will be either base or order
	error_rates[bsi]['base'] = {} #keys will be cv
	error_rates[bsi]['order'] = {}
	for cv in cv_partitions:
		base_error_rates = error_rates[bsi]['base']
		order_error_rates = error_rates[bsi]['order']
		base_error_rates[cv] = {} #keys will be resource id
		order_error_rates[cv] = {} #keys will be resource id
		resources = current_bsi_dict[cv+"resources"]
		current_data = current_bsi_dict[cv+"data"]
		for i in range(len(resources)): #basically which column
			row_with_response = -1
			for j in range(len(current_data)): #j is which row in data
				if int(current_data[j][i]) > 0:
					row_with_response = j
			resource_id = resources[i]
			if resource_id not in base_error_rates[cv]:
				base_error_rates[cv][resource_id] = []
			if resource_id not in order_error_rates[cv]:
				order_error_rates[cv][resource_id] = []	
			response = current_data[row_with_response][i]
			if response == "2":
				base_error_rates[cv][resource_id].append(1-float(current_bsi_dict[cv+"basepredictions"][row_with_response][i]))
				order_error_rates[cv][resource_id].append(1-float(current_bsi_dict[cv+"orderpredictions"][row_with_response][i]))
			else: 
				base_error_rates[cv][resource_id].append(float(current_bsi_dict[cv+"basepredictions"][row_with_response][i]))
				order_error_rates[cv][resource_id].append(float(current_bsi_dict[cv+"orderpredictions"][row_with_response][i]))

count_sig = 0
count_insig = 0
significant_results = []
sig_track = {} #first by bsi then by cv then by resource id
all_int_resources = []
all_error_rates = {}
all_base = {}
all_order = {}

for bsi in bsi_dict:
	# if bsi in ['6181', '7023', '6896', '7184']: #skipped to suppress ttest errors
	# 	continue
	current_bsi_dict = bsi_dict[bsi]
	sig_track[bsi] = {}
	all_base[bsi] = {}
	all_order[bsi] = {}
	int_resources = [int(n) for n in current_bsi_dict["c1resources"]]
	resource_count = max(int_resources)
	for i in range(2, resource_count+1):
		base_errors = []
		order_errors = []
		resource_id = str(i)
		for cv in cv_partitions:
			if resource_id not in error_rates[bsi]['base'][cv]:
				continue
			base_errors.extend(error_rates[bsi]['base'][cv][resource_id])
			order_errors.extend(error_rates[bsi]['order'][cv][resource_id])
		all_base[bsi][resource_id] = base_errors #ALL BASE ERRORS
		all_order[bsi][resource_id] = order_errors
		sig_track[bsi][resource_id] = ttest_rel(base_errors, order_errors)

significant = 0
insig = 0
sigs = []
for bsi in sig_track:
	for resource_id in sig_track[bsi]:
		if int(resource_id) not in relevant_resources[bsi]:
			continue
		if sig_track[bsi][resource_id][1] <= .05:
			significant += 1
			sigs.append([bsi, resource_id])
		else:
			insig += 1

base_better = 0
order_better = 0
better_bsis = set()
better_order = []
better_base = []
good_errors = []
base_errors = []
for bsi, rid in sigs:
	if int(rid) not in relevant_resources[bsi]:
		continue
	if mean(all_base[bsi][rid]) > mean(all_order[bsi][rid]):
		better_bsis.add(bsi)
		order_better += 1
		better_order.append([bsi, rid])
		good_errors.append(mean(all_order[bsi][rid]) - mean(all_base[bsi][rid]))
	else:
		base_better += 1
		better_base.append([bsi, rid])
		base_errors.append(mean(all_base[bsi][rid]) - mean(all_order[bsi][rid]))