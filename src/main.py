import argparse
from data_preprocess.process import *
from classifier import *
import pandas as pd
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', required=True,
                        help='Specify the mode of program')

    parser.add_argument('-f', '--input_file',help='input_file')

    # Parse the command-line arguments
    args = parser.parse_args()
    #Read the input files
    with open(args.input_file, 'r') as json_file:
        data = json.load(json_file)
    #Do data preprocess
    df = construct_dataset(data['dataset_path'])
    if(args.mode == "class"):
        df = add_pop_class(df)
    elif(args.mode == "cluster"):
        pass