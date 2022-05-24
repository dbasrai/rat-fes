from src.cort_processor import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass in folder')
    return parser.parse_args()

def main():
    args = parse_args()
    session = CortProcessor(args)

if __name__ == '__main__':
    main()
