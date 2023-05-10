import glob
import json
import cv2


# TEKNOFEST Log'larından labelları almak için yazılmış, çok işlevsel olmayabilir...
for i in glob.glob('*.log'):
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    for j in range(len(lines)):
        if (not lines[j].startswith('	{"frame": ')):
            continue
        frame = lines[j - 2][-17:].replace('\n', '')
        image = cv2.imread(frame)

        data = json.loads(lines[j].strip())['detected_objects']
        for k in data:
            start_point = (int(float(k['top_left_x'])),
                           int(float(k['top_left_y'])))
            end_point = (int(float(k['bottom_right_x'])),
                         int(float(k['bottom_right_y'])))

            if (k['cls'][-2] == '1'):
                color = (0, 255, 0)
                object_name = 'Tasit'
            elif (k['cls'][-2] == '2'):
                color = (0, 0, 255)
                object_name = 'Insan'
            elif (k['cls'][-2] == '3'):
                color = (0, 255, 255)
                object_name = 'UAP'
            else:
                color = (255, 0, 0)
                object_name = 'UAI'
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.putText(image, object_name, (int(float(k['top_left_x'])), int(
                float(k['top_left_y'])) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imwrite(frame, image)
