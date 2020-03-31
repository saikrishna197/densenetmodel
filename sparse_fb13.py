import cv2
import numpy as np
import argparse, time, math
import psutil, os, imutils
from skimage import measure
from keras.preprocessing import image
from keras.models import load_model
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help = "path to the model")
#ap.add_argument("-f", "--frames", default = 1, help = "number of frames to consider to make a decision of object in the frame")
ap.add_argument("-s", "--skip", default = 0, help = "number of frames to skip in between")
args = vars(ap.parse_args())


def class_detection(frame):
  global list_of_frames
  img = cv2.resize(frame, (224, 224))
  #converting the image to array
  x = image.img_to_array(img)
  # expanding the dimensions from (224,224,3) to (1, 224, 224, 3)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])

  #Predicting the result
  classes = model.predict(images)
  #print(classes)
  #mapping = {'Bookpagetext': classes[0][0], 'CostumeParty': classes[0][1], 'JrkitABC': classes[0][2], 'Newton': classes[0][3], 'SquigleMagic': classes[0][4], 'Tangram': classes[0][5], 'WorksheetPage': classes[0][6]}
  #mapping = {'book_cover' : classes[0][0], 'book_page':classes[0][1], 'book_page_graphics':classes[0][2], 'jrkitfaces': classes[0][3], 'jrkitsticks':classes[0][4], 'tangram':classes[0][5], 'worksheet_page':classes[0][6]}
  mapping = {'book_cover' : classes[0][0], 'book_page':classes[0][1], 'book_page_graphics':classes[0][2], 'jrkitfaces': classes[0][3], 'jrkitfaces_box' : classes[0][4], 'jrkitsticks':classes[0][5], 'jrkitsticks_box' : classes[0][6], 'tangram':classes[0][7], 'tangram_box' : classes[0][8], 'worksheet_cover':classes[0][9], 'worksheet_page' : classes[0][10]}
  result_each_frame = sorted(mapping.items(), key = lambda i: i[1], reverse=True)
  #appending the detection to list_of_frames list
  #list_of_frames.append(result_each_frame[0])

  # args['frames'] is the number of frames we need to take average upon. So if the len of list_of_frames increases by that the first frame will be popped out.
  # if len(list_of_frames) > args['frames']:
  #     list_of_frames.pop(0)

  #calling the classifying function where average results computation occurs.
 # average_label = classifying(list_of_frames)
  #print("RESULT EACH FRAME", result_each_frame[0])
  #print("Average Label:", average_label)
  #print("result each frame", result_each_frame)
  return result_each_frame


def frame_difference(label, frame):
    global different_frames
    #frame = cv2.normalize(frame,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    frame = cv2.resize(frame, (500, 500))
    frame = frame[30:380, 40:460]
    frame = cv2.cvtColor(cv2.resize(frame, (300, 300)), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("first", first_frame)
    # cv2.imshow("second", second_frame)
    # gradient_1 = get_it_converted(first_frame)
    # gradient_2 = get_it_converted(second_frame)
    # print(f"Gradient 1 : {gradient_1[1:]}")
    # print(f"Gradient 2 : {gradient_2[1:]}")
    #laplace = cv2.Laplacian(second_frame, cv2.CV_64F).var()
    print("Length of different frames list : ", len(different_frames))
    #second_frame = second_frame[40:300, 20:380]
    if len(different_frames) == 0:
        different_frames.append((frame, count))
        cv2.imwrite(f"testing/stable_different_frame/{count}.jpg", frame)
        # print(different_frames)
        # print(different_frames[-1][1])
    if len(different_frames) == 3:
        different_frames.pop(0)
    if len(different_frames) == 2 or len(different_frames) == 1:
        print("length is 2 now")
        first_frame = different_frames[-1][0]
        first_frame_canny = cv2.Canny(first_frame, 100, 200)
        frame_canny = cv2.Canny(frame, 100, 200)
        orb = cv2.ORB_create()
        h1= np.mean(frame)
        h2 = np.mean(first_frame)
        h3 = abs(h1-h2)
        print(h3)
        kp1, des1 = orb.detectAndCompute(first_frame, None)
        kp2, des2 = orb.detectAndCompute(frame, None)
        if des1 is None or des2 is None:
            return "same"
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#113, 138, 159, 166, 328
        matches = bf.match(des1, des2)
        s = measure.compare_ssim(frame, first_frame)
        absdiff_value = np.mean(cv2.absdiff(frame_canny, first_frame_canny))/(np.mean(frame_canny)+np.mean(first_frame_canny))
        print(f"FRAME NO : {count} and MATCHES : {len(matches)} and SSIM : {s} and ABSDIFF VALUE : {absdiff_value}")
        print(f"COMPARING FRAME NOS : {count} and {different_frames[-1][1]}")
        if label[0] in ["book_cover", "book_page", "book_page_graphics"]:
            
            if absdiff_value >= 0.78:
                print("inside absdiff")
                if s < 0.7:
                    print("inside ssim")
                    if len(matches) < 149:#300:
                        print(f"COMPARING FRAME NOS : {count} and {different_frames[-1][1]}")
                        print(f"===inside frame difference : {len(matches)} and ssim : {s} and absdiff : {absdiff_value}===")
                        #if gradient_2[1] >= 1.0:
                        cv2.imwrite(f"testing/stable_different_frame/{count}.jpg", frame)
                        different_frames.append((frame, count))
                        return "different"
                    else:
                        return "same"
                else:
                    return "same"
            else:
                return "same"
        elif label[0] in ["worksheet_page", "worksheet_cover"]:
            if absdiff_value >= 0.7:
                print("inside absdiff")
                if s < 0.7:
                    print("inside ssim")
                    if len(matches) < 149:#300:
                        print(f"COMPARING FRAME NOS : {count} and {different_frames[-1][1]}")
                        print(f"===inside frame difference : {len(matches)} and ssim : {s} and absdiff : {absdiff_value}===")
                        #if gradient_2[1] >= 1.0:
                        cv2.imwrite(f"testing/stable_different_frame/{count}.jpg", frame)
                        different_frames.append((frame, count))
                        return "different"
                    else:
                        return "same"
                else:
                    return "same"
            else:
                return "same"
        elif label[0] in ["tangram", "tangram_box", "jrkitfaces", "jrkitfaces_box", "jrkitsticks", "jrkitsticks_box"]:
            if round(absdiff_value, 2) > 0.4:
                print(f"COMPARING FRAME NOS : {count} and {different_frames[-1][1]}")
                print("absdiff crossed")
                if s < 0.7:
                    print("ssim crossed")
                    if len(matches) < 126:
                        #if gradient_2[1] >= 1.0:
                        different_frames.append((frame, count))
                        cv2.imwrite(f"testing/stable_different_frame/{count}.jpg", frame)
                        return "different"
                    else:
                        return "same"
                else:
                    return "same"
            else:
                return "same"


def get_it_converted(frame):
    frame = cv2.resize(frame, (300, 300))
    frame = frame[10:250, 30:280]
    ddepth = cv2.CV_64F
    kernel_size = 3
    #gray_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray_1, (3,3), 0)

    sch_x = cv2.Scharr(frame, ddepth, 1, 0)
    sch_y = cv2.Scharr(frame, ddepth, 0, 1)

    abs_sch_x = cv2.convertScaleAbs(sch_x)
    abs_sch_y = cv2.convertScaleAbs(sch_y)
    magnitude = round(math.sqrt(np.sum(np.add(sch_x*sch_x, sch_y*sch_y)))/90000, 1)
    magnitude_x = np.sum(sch_x)/90000
    magnitude_y = np.sum(sch_y)/90000
    grad_total = cv2.addWeighted(abs_sch_x, 0.5, abs_sch_y, 0.5, 0)
    #angle = cv2.arctan2(abs_sch_y, abs_sch_x)
    return grad_total, magnitude, magnitude_x, magnitude_y


cap = cv2.VideoCapture("priority_one.3gp")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('test_all_items.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
stables = []
different_frames = list()
frame_no=10
flag_set=0
# initialize the first frame in the video stream
firstFrame = None
frames = []
count=0
labels = []
model = load_model('vgg16-try-1_f12.h5',compile=False)
# loop over the frames of the video
while cap.isOpened():
    
    # grab the current frame and initialize the occupied/unoccupied
    # text
    _, frame = cap.read()
    if frame is None:
        print("Video Ended")
        break
    frame_store = frame.copy()
    overlay = frame.copy()
    output = frame.copy()
    #print(frame)
    start = time.time()
    #print(frame)
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    frame_no += 1

    if args["skip"] == 0 or frame_no == args["skip"]:
        frame_no = 0
        # resize the frame, convert it to grayscale, and blur it
        frame_diff = imutils.resize(frame, width=500)
        frame_diff = frame_diff[10:340, 70:420]
        h, w = frame.shape[:2]
        frame_stable = frame[int(0.4*h):int(0.75*h), int(0.18*w):int(0.68*w)]
        frame_stable = cv2.resize(frame_stable, (300, 300))
        #frame_motion = frame_stable.copy()
        mean=np.mean(frame_stable)
        #print("mean is ",mean)

        dil = cv2.dilate(frame_stable, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dil, 51)
        diff_img = 255 - cv2.absdiff(frame_stable, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)

        #cv2.imshow("result norm", result_norm)
        #cv2.waitKey(0)
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)

        # if the first frame is None, initialize it
        if firstFrame is None:
            frames.append(gray)
            firstFrame = 1
            count += 1
            continue
        else:
            firstFrame = frames[0]
            frames.clear()
            frames.append(gray)
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        if mean > 200:
            mean -= 60
        thresh = cv2.threshold(frameDelta, mean, 255, cv2.THRESH_BINARY)[1]
        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnt=cont_(contours)

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        #thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        #cv2.imshow("after erode + dilate", thresh)
        sum_pixels = np.sum(thresh)
        #print("Time uppar uppar vaala : ", time.time() - start)
        #print(sum_pixels)
        print("Sum of the frameDelta difference : ", np.sum(frameDelta), count)
        if np.sum(frameDelta) > 500000: #and cnt>=1:
            cv2.imwrite(f"testing/motion_test5/{count}_{mean}.jpg", frame)
            #print(frame_stable.shape)
            #print("Motion Detected")
            #cv2.putText(frame, "Motion Detected", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #out.write(cv2.resize(frame, (frame_width, frame_height)))
            #time.sleep(2)
            #cv2.imwrite(os.path.join(path, str(count)+".jpg"), frame)
        else:
            #print(frame_stable.shape)
            if len(stables) > 2:
                stables.pop(0)
            # # cv2.imwrite("one.jpg", gray)
            # # cv2.imwrite("two.jpg", firstFrame)

            stables.append(frame)
            #print("No Motion Detected")
            #cv2.putText(frame, "No Motion Detected", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # #print(1)
            # #start = time.time()
            

            label = class_detection(frame)
            print(label[0])
            # #print("class detected")
            if label[0][0] == "empty":
                print("Empty class detected")
                count += 1
                #out.write(cv2.resize(frame, (frame_width, frame_height)))
                continue
            cv2.imwrite(f"testing/cropped_no_motion/{count}_{label[0][0]}_{label[0][1]}.jpg", frame_stable)
            cv2.imwrite(f"testing/no_motion_test5/{count}_{label[0][0]}_{label[0][1]}.jpg", frame)
            if label[0][1] < 0.72:
                print("less than 70% confidence on detection", label[0][0], label[0][1])
                count += 1
                #out.write(cv2.resize(frame, (frame_width, frame_height)))
                continue
            print("Threshold condition crossed")
            cv2.imwrite(f"testing/no_motion_more_confidence/{count}_{label[0][0]}_{label[0][1]}.jpg", frame)
            # #print("Time taken for class detection : ", time.time() - start)
            labels.append(label[0])
            alpha = 0.4
            # #print("Labels : ", labels)
            print("Length of labels : ", len(labels))
            if len(labels) > 2:
                labels.pop(0)
            #print("Labels after : ", labels)
            if len(labels) == 1:
                print("length of labels is 1")
                print(f"Class Opened : {labels[0][0]}")
            if len(labels) == 2:
                overlay = frame.copy()
                if not labels[0][0] == labels[1][0]:
                    print("Class opened: ", labels[1][0])
                    cv2.rectangle(overlay, (30, 10), (720, 50), (0, 0, 0), -1)
                    cv2.putText(overlay, f"different class detected {labels[1][0]}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    cv2.imwrite(f"testing/different_class/{count}_{labels[1][0]}__{labels[1][1]}.jpg", frame)
                    cv2.imshow(f"different", frame)
                else:
                    print(1)
                    frame_different = frame_difference(label[0], stables[1])
                    print(2)
                    if frame_different == "same":
                        print()
                        #cv2.putText(frame, "same frame", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        #cv2.imwrite(f"testing/same_frame_same_class/{count}_{labels[1][0]}__{labels[1][1]}.jpg", frame)
                    else:
                        cv2.rectangle(overlay, (30, 10), (720, 50), (0, 0, 0), -1)
                        print("Class opened : ", labels[1][0], labels[1][1])
                        cv2.putText(frame, f"different frame with {labels[1][0]} class detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.imshow(f"different", frame)
                        cv2.imwrite(f"testing/different_frame_same_class/{count}___{labels[1][0]}__{labels[1][1]}.jpg", frame)
        #print("TIME TAKEN PER FRAME : ", time.time() - start)
        #out.write(cv2.resize(frame, (frame_width, frame_height)))
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
        if key == ord("f"):
            flag_set = 1
        if key == ord("r"):
            flag_set = 0
    else:
        count += 1
        continue
    count+=1
# cleanup the camera and close any open windows
print("At the end length of the list : ", len(different_frames))
cv2.imwrite(f"testing/stable_different_frame/{different_frames[-1][1]}.jpg", different_frames[-1][0])
print("Releasing the camera")
cap.release()
#out.release()
print("Destroying All Windows")
cv2.destroyAllWindows()