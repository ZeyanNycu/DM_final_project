import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', required=True,
                        help='Specify the mode of program')

    parser.add_argument('-f', '--input_file',help='input_file')

    # Parse the command-line arguments
    args = parser.parse_args()
    #Do data preprocess
    if(args.mode == "class"):
        pass
    elif(args.mode == "cluster"):
        pass