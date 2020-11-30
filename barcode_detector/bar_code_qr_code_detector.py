import pyzbar.pyzbar as pyzbar
import cv2
import matplotlib.pyplot as plt

n=10
f=open("inform.txt", "w")

for i in range(1, n+1):
    qr_code_img=cv2.imread('./img/img{}.jpg'.format(i))
    f.write("[img{}]".format(i))
    
    print('img read......')
    plt.imshow(qr_code_img)

    print('convert gray scale....')
    gray=cv2.cvtColor(qr_code_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')

    decoded=pyzbar.decode(gray)
    print(decoded)

    for d in decoded:
        f.write(" type"+d.type+" : "+d.data.decode('utf-8')+"\n")
        cv2.rectangle(qr_code_img, (d.rect[0], d.rect[1]), (d.rect[0]+d.rect[2], d.rect[1]+d.rect[3]), (255, 0, 0), 1)

f.close()

#qr-code making site --> 1. https://qr.naver.com/
#                        2. https://ko.qr-code-generator.com/

#ing....
