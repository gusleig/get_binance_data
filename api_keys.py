from configparser import ConfigParser
import os.path
import sys
from dotenv import load_dotenv

config_object = ConfigParser()

load_dotenv()

api_key = os.getenv('api_key')
api_secret = os.getenv('api_secret')

if not api_key or not api_secret:
    if not os.path.isfile("config.ini"):
        print("Check config file config.ini for api key parameters")

        config_object.add_section('API_CONFIG')
        config_object.set('API_CONFIG', 'api_key', 'xxxx')
        config_object.set('API_CONFIG', 'api_secret', 'xxxx')

        with open(r"config.ini", 'w') as configfile:
            config_object.write(configfile)

        sys.exit()

    config_object.read("config.ini")

    api_key = config_object['API_CONFIG']['api_key']
    api_secret = config_object['API_CONFIG']['api_secret']

