import json

NAME = "config_11_breast_cancer_radial_basis"

input_file = NAME + ".json"
output_file = NAME + ".csv"

with open(input_file) as json_file:
    config_json = json.load(json_file)

csv_table = ["key,value"]
for k, v in config_json.items():
    csv_table.append(",".join([k, str(v)]))

with open(output_file, 'w') as f:
    f.write("\n".join(csv_table))
