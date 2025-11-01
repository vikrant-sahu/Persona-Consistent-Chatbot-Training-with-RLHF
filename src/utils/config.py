import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_config(config_name):
    config_path = f'configs/{config_name}.yaml'
    return load_config(config_path)