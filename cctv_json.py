import json as js
import os
import cv2

json_path = '/run/media/michael/SSD2/cctv_fight_dataset/dataset/ground-truth.json'
fi_savedir = '/home/michael/AVSS2019/src/VioDB/cctv_jpg/fi/'
no_savedir = '/home/michael/AVSS2019/src/VioDB/cctv_jpg/no/'
video_dir = '/run/media/michael/SSD2/cctv_fight_dataset/dataset/'
cctv_json = {}

def read_json(file_path):
    with open(file_path, 'r',
              encoding='utf8')as fp:
        json_data = js.load(fp)
        return json_data
        data = json_data['data']


def create_json(video_name,subset,annotations):
    annotations_json = {'label':annotations}
    video_name_json = {'subset':subset,'annotations':annotations_json}
    cctv_json[video_name] = video_name_json


def convert_cctv_dataset():
    database = read_json(json_path)['database']
    count = 0
    for video_name in database:
        subset = database[video_name]['subset']
        source = database[video_name]['source']
        fps = database[video_name]['frame_rate']
        if source == 'CCTV':
            source = 'CCTV_DATA'
        else:
            source = 'NON_CCTV_DATA'
        video_path = video_dir + source + '/' + subset + '/' + video_name + '.mpeg'
        annotations = database[video_name]['annotations']
        video = cv2.VideoCapture(video_path)
        # video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


        fi = no = 1
        fi_begin = int(annotations[0]['segment'][0] * fps)
        fi_end = int(annotations[0]['segment'][1] * fps)
        if fi_end - fi_begin > 350:
            fi_end = fi_begin + 350
        if fi_begin - (fi_end-fi_begin) < 0:
            frame_count = 0
        else:
            frame_count = fi_begin - (fi_end-fi_begin)

        fi_save_path = fi_savedir + video_name + '_fi/'
        no_save_path = no_savedir + video_name + '_no/'
        if not os.path.exists(fi_save_path):
            os.makedirs(fi_save_path)
        if not os.path.exists(no_save_path):
            os.makedirs(no_save_path)


        create_json(video_name + '_fi',subset,'fi')
        create_json(video_name + '_no',subset,'no')

        while frame_count < fi_end:
            res,image = video.read()
            image = cv2.resize(image,(320,240),)
            if not res:
                print('not res , not image')
                break
            if frame_count < fi_begin: # normal
                name_count = "{:05}".format(no)
                cv2.imwrite(no_save_path + 'image_' + name_count + '.jpg',image)
                print(no_save_path + 'image_' + name_count + '.jpg')
                no += 1
            elif frame_count < fi_end: # violence
                name_count = "{:05}".format(fi)
                cv2.imwrite(fi_save_path + 'image_' + name_count + '.jpg',image)
                print(fi_save_path + 'image_' + name_count + '.jpg')
                fi += 1
            frame_count += 1
        f = open(no_save_path + 'n_frames',"x")
        f.write(str(no-1))
        f.close()
        f = open(fi_save_path + 'n_frames',"x")
        f.write(str(fi-1))
        f.close()
        count = count +1
        if count == 20:
            break;

def main():
    convert_cctv_dataset()
    cctv_jpg = {'labels':['fi','no'],'database':cctv_json}
    cctv_jpg_json = js.dumps(cctv_jpg)
    f2 = open('cctv_jpg.json', 'w')
    f2.write(cctv_jpg_json)
    f2.close()

if __name__ == '__main__':
    main()
