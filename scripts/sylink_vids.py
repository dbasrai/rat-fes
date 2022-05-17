import os

def doit(folder):
    substring = 'live_videos'
    vid_name_list = []
    vid_path_list= []
    for root, dirs, files in os.walk(folder):
        if substring in root:
            for filename in files:
                if filename.endswith('.avi'):
                    vid_name_list.append(f'videos/{filename}')
                    vid_path_list.append(f'{root}/{filename}')
    os.mkdir('videos')

    for i in range(len(vid_name_list)):
        os.symlink(vid_path_list[i], vid_name_list[i])

