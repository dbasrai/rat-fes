import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='pass in folder')
    parser.add_argument('echo')
    return parser.parse_args()


def main():
    args = parse_args()
    substring = 'live_videos'
    vid_name_list = []
    vid_path_list= []
    for root, dirs, files in os.walk(args.echo):
        if substring in root:
            for filename in files:
                if filename.endswith('.avi'):
                    print(filename)
                    vid_name_list.append(f'videos/{filename}')
                    vid_path_list.append(f'{root}/{filename}')
    os.mkdir('./videos')

    for i in range(len(vid_name_list)):
        os.symlink(vid_path_list[i], vid_name_list[i])

if __name__ == '__main__':
    main()
