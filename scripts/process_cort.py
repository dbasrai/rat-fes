from src.cort_processor import *
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass in folder')
    parser.add_argument('echo')
    return parser.parse_args()

def main():
    args = parse_args()
    session = CortProcessor(args.echo)
    session.process()
    animal_name = session.handler.folder_path.split('/')[-2]
    session_name = session.handler.folder_path.split('/')[-1]
    with open(f'/home/diya/Documents/rat-fes/data/pickles/{animal_name}_{session_name}_session.pkl', 'wb') as inp:
        pickle.dump(session, inp)

if __name__ == '__main__':
    main()
