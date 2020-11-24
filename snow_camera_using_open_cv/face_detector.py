import cv2, dlib, sys
import numpy as np

scaler=0.2

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap=cv2.VideoCapture('faces.mp4')
santa=cv2.imread('santa_1.png', cv2.IMREAD_UNCHANGED)

fourcc=cv2.VideoWriter_fourcc(*'DIVX')

w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
f=cap.get(cv2.CAP_PROP_FPS)

print("width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps:", cap.get(cv2.CAP_PROP_FPS))

out=cv2.VideoWriter('./save_video/save.avi', fourcc, f, (int(w*scaler), int(h*scaler)))

#santa_ overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    
    bg_img = background_img.copy()
    
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
      img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img

face_roi=[]
face_sizes=[]

#video loop
while True:
    ret, img=cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    origin_img=img.copy()

    #Isfaces
    if len(face_roi)==0:
        faces=detector(img, 1)
    else:
        roi_img=img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        faces=detector(roi_img)

    if len(faces)==0:
        print("no faces!")

    for face in faces:
        if len(face_roi)==0:
            dlib_shape=predictor(img, face)
            shape_2d=np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape=predictor(roi_img, face)
            shape_2d=np.array([[p.x+face_roi[2], p.y+face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=1)

        #center
        c_x, c_y=np.mean(shape_2d, axis=0).astype(np.int)

        
        min_coords=np.min(shape_2d, axis=0)
        max_coords=np.max(shape_2d, axis=0)

        face_size=max(max_coords-min_coords)
        face_sizes.append(face_size)

        mean_face_size=int(np.mean(face_sizes)*1.8)

        # compute face roi
        face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
        face_roi = np.clip(face_roi, 0, 10000)
    
        result=overlay_transparent(origin_img, santa, c_x-10 , c_y-100, overlay_size=(mean_face_size, mean_face_size))
        cv2.circle(result, center=tuple((c_x, c_y)), radius=5, color=(0,0,255), thickness=10)
        
    #cv2.imshow('orginal_img', origin_img)    
    cv2.imshow('facial landmarks', img)
    cv2.imshow('result', result)
    out.write(result)
    
    if cv2.waitKey(1)==ord('q'):
        sys.exit(1)

cap.release()
out.release()
cv2.destroyAllWindows()
