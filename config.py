from __future__ import print_function

import os
import sys
import logging

if sys.version_info[0] < 3:
    import ConfigParser as configparser
else:
    import configparser

config = configparser.SafeConfigParser()

get = config.get
config_dir = os.path.dirname(__file__)

def parser_config():
    config_file = os.path.join(config_dir, "conf")

    if not os.path.exists(config_file):
        sys.stderr.write("Error: Unable to find the config file!\n")
        sys.exit(1)

    # parse the configuration
    global config
    config.readfp(open(config_file))


parser_config()

COMP = {
    "Permission": "permission",
    "Activity": "activity",
    "Service": "service",
    "Receiver": "receiver",
    "Provider": "provider",
    "Hardware": "hardware",
    "Intentfilter": 'intent-filter',
    "Android_API": "android_api",
    "Java_API": "java_api",
    "User_String": "const-string",
    "User_Class": "user_class",
    "User_Method": "user_method",
    "OpCode": "opcode",
    "Asset": "asset",
    "Notdefined": 'not_defined'
}

logging.basicConfig(level=logging.INFO, filename=os.path.join(config_dir, "log"), filemode="w",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setLevel(logging.WARNING)
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))
