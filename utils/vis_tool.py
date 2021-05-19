import cv2
import os
import sys
import numpy as np
from dataset.config import BF_ACTION_CLASS, BF_ACTION_COLOR

class VISUALIZER(object):
    def __init__(self, v_dir=None):
        self.v_dir = v_dir
        self.record_img = None
        self.offset = 130
        self.resize_ratio = 2
        self.bar_height = 20
        self.bar_left_offset = 20
        self.bar_top_offset = 30
        self.fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.fontScale = 1
        self.fontThickness = 1
        self.lineThickness = 1

    def show(self, f_name, gt_labels_list, recog_res, anti_res, obs_prec):
        video_dir = os.path.join(*f_name.replace('rgb_frame', 'original').split('/')[:-1])
        video_dir = os.path.join(video_dir, [x for x in os.listdir(video_dir) if f_name.split('/')[-1] in x and 'labels' not in x][0])
        capture = cv2.VideoCapture(video_dir)
        if not capture.isOpened():
            print("Failed to open the video!!!")
            sys.exit(1)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        fcount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fwidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))*self.resize_ratio
        fheight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))*self.resize_ratio

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(self.v_dir+f'/{obs_prec}.avi', fourcc, fps, (fwidth, fheight+self.offset))
        cv2.namedWindow(f_name.split('/')[-1])
        cv2.moveWindow(f_name.split('/')[-1], 50, 50)

        unit = 600 // round(fcount/fps+0.5)
        sec = 0
        self.record_img = None; mark=True
        for i in range(fcount):
            ret, frame = capture.read()
            if not ret and frame is None:
                continue
            frame = cv2.resize(frame, (fwidth, fheight))[:,:,::-1]
            if self.record_img is None:
                self.record_img = np.zeros((fheight+self.offset, fwidth, 3), dtype=np.uint8)
            self.record_img[:fheight, :fwidth, :] = frame
            # plot every second
            if i % fps == fps-1:
                # import ipdb; ipdb.set_trace()
                sec += 1
                # use the main action as the label in a second
                gt_label = gt_labels_list[(sec-1)*fps+fps//2]
                gt_text = f'GT:{gt_label}'
                # for gt bar
                self.record_img[fheight+1:fheight+self.bar_top_offset,:,:] = 0
                cv2.putText(self.record_img, gt_text, (self.bar_left_offset, fheight+self.bar_top_offset-10), fontScale=self.fontScale, fontFace=self.fontFace, thickness=self.fontThickness, color=(255,255,255))
                cv2.rectangle(self.record_img, (self.bar_left_offset+(sec-1)*unit, fheight+self.bar_top_offset), (self.bar_left_offset+sec*unit, fheight+self.bar_top_offset+self.bar_height), thickness=-1, color=BF_ACTION_COLOR[gt_label])
                if sec <= len(recog_res):
                    pred_label = BF_ACTION_CLASS[recog_res[sec-1]]
                    pred_text = f'Reco_pred:{pred_label}'
                    # for pred bar
                    cv2.rectangle(self.record_img, (self.bar_left_offset+(sec-1)*unit, fheight+self.bar_top_offset*2+self.bar_height), (self.bar_left_offset+sec*unit, fheight+(self.bar_height+self.bar_top_offset)*2), thickness=-1, color=BF_ACTION_COLOR[pred_label])                
                else:
                    s_idx = len(recog_res)
                    if sec <= len(anti_res)+s_idx:
                        pred_label = BF_ACTION_CLASS[anti_res[sec-s_idx-1]]
                        pred_text = f'Anti_pred:{pred_label}'
                        cv2.rectangle(self.record_img, (self.bar_left_offset+(sec-1)*unit, fheight+self.bar_top_offset*2+self.bar_height), (self.bar_left_offset+sec*unit, fheight+(self.bar_height+self.bar_top_offset)*2), thickness=-1, color=BF_ACTION_COLOR[pred_label])
                    else:
                        pred_text = 'Anti_pred:None'
                    if mark:
                        cv2.line(self.record_img, (self.bar_left_offset+(sec-1)*unit, fheight+self.bar_top_offset), (self.bar_left_offset+(sec-1)*unit, fheight+(self.bar_height+self.bar_top_offset)*2), (255,255,255), self.lineThickness) 
                        cv2.putText(self.record_img, obs_prec+'obs', (max(self.bar_left_offset+(sec-1)*unit-80, 0), fheight+self.bar_height+self.bar_top_offset*2-10), fontScale=self.fontScale, fontFace=self.fontFace, thickness=self.fontThickness, color=(255,255,255))
                        cv2.putText(self.record_img, '0.5anti', (self.bar_left_offset+(sec-1)*unit+8, fheight+self.bar_height+self.bar_top_offset*2-10), fontScale=self.fontScale, fontFace=self.fontFace, thickness=self.fontThickness, color=(255,255,255))
                        mark=False
                self.record_img[fheight+(self.bar_top_offset+self.bar_height)*2+1:,:,:]=0
                cv2.putText(self.record_img, pred_text, (self.bar_left_offset, fheight+(self.bar_top_offset+self.bar_height)*2+20), fontScale=self.fontScale, fontFace=self.fontFace, thickness=self.fontThickness, color=(255,255,255))                
            # show the results
            cv2.imshow(f_name.split('/')[-1], np.array(self.record_img)[:,:,::-1])
            video.write(np.array(self.record_img)[:,:,::-1])

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()
