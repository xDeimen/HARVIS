import yaml

def read_yml(yml_path):
    with open(yml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":
    read_yml("./.models/config.yml")