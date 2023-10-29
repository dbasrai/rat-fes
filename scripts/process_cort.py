from src.cort_processor import *
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass in folder')
    parser.add_argument('echo')
    # parser.add_argument('thresh', type=float)
    return parser.parse_args()

def main():
    args = parse_args()
    pickles_dir = ("./../data/pickles/automatic2.1_norefrac")
    CHECK_FOLDER = os.path.isdir(pickles_dir)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(pickles_dir)
        print("created folder : ", pickles_dir)

    else:
        print(pickles_dir, "folder already exists.")
    nest = args.echo
    days = ['923','1004','1010','1017']
    for i in range(len(days)):
        session = CortProcessor(f'{nest}/{days[i]}')
        session.process()
        animal_name = session.handler.folder_path.split('/')[-2]
        session_name = session.handler.folder_path.split('/')[-1]
        with open(f'{pickles_dir}/{animal_name}_{session_name}_session.pkl', 'wb') as inp:
            pickle.dump(session, inp)
    

if __name__ == '__main__':
    main()
# def main():
#     args = parse_args()
#     pickles_dir = ("./../data/pickles/automatic2.1_norefrac")
#     CHECK_FOLDER = os.path.isdir(pickles_dir)

#     # If folder doesn't exist, then create it.
#     if not CHECK_FOLDER:
#         os.makedirs(pickles_dir)
#         print("created folder : ", pickles_dir)

#     else:
#         print(pickles_dir, "folder already exists.")
#     session = CortProcessor(args.echo)
#     session.process()
#     animal_name = session.handler.folder_path.split('/')[-2]
#     session_name = session.handler.folder_path.split('/')[-1]
#     with open(f'{pickles_dir}/{animal_name}_{session_name}_session.pkl', 'wb') as inp:
#         pickle.dump(session, inp)
