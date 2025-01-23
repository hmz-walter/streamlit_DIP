import cv2
import os
import glob
import numpy as np
DEBUG=False


default_params = {
    'scoreThreshold': 50,
    'reliabilityThreshold': 50,
    'centerDistanceThreshold': 10,
    'blur': 5,
    'min_thresh': 33,
    'max_ratio': 100,
    'error_ratio': 10,
    'clipLimit': 1,
    'tileGridSize': 0,
    'C': 0,
    'blocksize': 0
}

if DEBUG:
    # create a windows to change params
    cv2.namedWindow('param', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('scoreThreshold', 'param', 50, 100, lambda x: None)
    cv2.createTrackbar('reliabilityThreshold', 'param', 50, 100, lambda x: None)
    cv2.createTrackbar('centerDistanceThreshold', 'param', 10, 100, lambda x: None)
    cv2.createTrackbar('blur', 'param', 5, 50, lambda x: None)
    # canny
    cv2.createTrackbar('min_thresh', 'param', 33, 100, lambda x: None)
    cv2.createTrackbar('max_ratio', 'param', 100, 100, lambda x: None)

    cv2.createTrackbar('error_ratio', 'param', 10, 100, lambda x: None)


    # clipLimit
    cv2.createTrackbar('clipLimit', 'param', 1, 10, lambda x: None)
    # tileGridSize
    cv2.createTrackbar('tileGridSize', 'param', 0, 100, lambda x: None)

    cv2.createTrackbar('C', 'param', 0, 100, lambda x: None)
    cv2.createTrackbar('blocksize', 'param', 0, 100, lambda x: None)

def get_params(name):
    if DEBUG:
        return get_params(name, 'param')
    else:
        return default_params[name]

def imshow(name, img):
    if DEBUG:
        cv2.imshow(name, img)
        

def iou(ellipse1, ellipse2):
    x,y,a,b,radius,score=ellipse1.flatten()
    x2,y2,a2,b2,radius2,score2=ellipse2.flatten()
    
    # calculate iou as rotated rectangle
    # calculate the overlap area
    box1=cv2.RotatedRect((x,y),(a,b),radius)
    box2=cv2.RotatedRect((x2,y2),(a2,b2),radius2)
    intersection=cv2.rotatedRectangleIntersection(box1, box2)
    if intersection[0]==0:
        return 0
    else:
        intersection=intersection[1]
        intersection_area=cv2.contourArea(intersection)
        area1=cv2.contourArea(box1.points())
        area2=cv2.contourArea(box2.points())
        return intersection_area/(area1+area2-intersection_area)
        
    

def nms(ellipses:np.ndarray,threshold=0.2):
    # sort by score
    indices=np.argsort(ellipses[:,0,5])[::-1]
    keep=[]
        
    while len(indices) > 0:
        # 选择得分最高的边界框
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        # 计算IoU
        ious = np.array([iou(ellipses[current], ellipses[i]) for i in indices[1:]])

        # 保留IoU小于阈值的边界框
        indices = indices[1:][ious < threshold]

    return ellipses[keep,:]

    

def det_coin(img):
    
    
    # img shape
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    max_width = 640
    if img_width > max_width:
        img = cv2.resize(img, (max_width, int(max_width/img_width*img_height)))
        img_width = img.shape[1]
        img_height = img.shape[0]
    
    # do auto white balance
    result = cv2.xphoto.createSimpleWB()
    result.setP(1)
    img = result.balanceWhite(img)
    
    
    
    
    min_thresh=min(img_width,img_height)/10
    max_thresh=min(img_width,img_height)/2
    
    # guassian blur
    blur = get_params('blur')*2+1
    #blur = 5
    #img = cv2.bilateralFilter(img, 7, 75, 75)
    img = cv2.GaussianBlur(img, (blur,blur), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clipLimit = get_params('clipLimit')
    tileGridSize = get_params('tileGridSize')+1
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    gray = clahe.apply(gray)
    imshow('gray', gray)
    #gray = cv2.bilateralFilter(gray, 7, 75, 75)


    min_thresh = get_params('min_thresh',)
    max_ratio = get_params('max_ratio')/20
    max_thresh = min_thresh*max_ratio
    min_thresh, max_thresh = 50, 250
    edges = cv2.Canny(gray, min_thresh, max_thresh)
    imshow('edges', edges)
    
    
    # # canny edge detection
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, min_thresh, max_thresh)
    # imshow('edges', edges)
    scoreThreshold = get_params('scoreThreshold')/100
    reliabilityThreshold = get_params('reliabilityThreshold')/100
    centerDistanceThreshold = get_params('centerDistanceThreshold')
    ret=cv2.ximgproc.findEllipses(image=gray, ellipses=None,scoreThreshold=scoreThreshold,reliabilityThreshold=reliabilityThreshold,centerDistanceThreshold=centerDistanceThreshold )

    if ret is None:
        return None
    
    # filter out the ellipses 
    ret_a_b_ratio = ret[:,0,2]/ret[:,0,3]
    median_a_b_ratio = np.median(ret_a_b_ratio)
    median_a=np.median(ret[:,0,2])
    median_b=np.median(ret[:,0,3])
    print('median a/b ratio:', median_a_b_ratio)
    # filter out ratio error than 0.2
    error_ratio = get_params('error_ratio')/100
    ori_shape = ret.shape
    mask=ret_a_b_ratio>median_a_b_ratio*(1-error_ratio)
    mask=mask & (ret_a_b_ratio<median_a_b_ratio*(1+error_ratio))
    # filter out a and b
    mask=mask & (ret[:,0,2]<median_a*1.5)
    mask=mask & (ret[:,0,3]<median_b*1.5)
    mask=mask & (ret[:,0,2]>median_a/1.5)
    mask=mask & (ret[:,0,3]>median_b/1.5)
    ret = ret[mask]
    filtered_shape = ret.shape
    filtered_data = ori_shape[0]-filtered_shape[0]
    print('filtered data:', filtered_data)
    ret=nms(ret)
    
    print('nms:', ret)
    
    five_index=find_five_mao(img,ret)
    avr_five_a=np.mean(ret[five_index,0,2])
    avr_five_b=np.mean(ret[five_index,0,3])
        
    sum_jiao=0
    for i_ellipse in range(ret.shape[0]):
        ellipse=ret[i_ellipse,0,:]
        x,y,a,b,radius,score=ellipse
        if i_ellipse in five_index:
            cv2.ellipse(img, (int(x),int(y)), (int(a),int(b)), int(radius), 0, 360, (0, 255, 255), 2)
            cv2.putText(img, f"{0.5}r" , (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            sum_jiao+=5
        else:
            if a>avr_five_a and b>avr_five_b:
                cv2.ellipse(img, (int(x),int(y)), (int(a),int(b)), int(radius), 0, 360, (255, 255, 0), 2)
                cv2.putText(img, f"{1}r", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                sum_jiao+=10
            else:
                cv2.ellipse(img, (int(x),int(y)), (int(a),int(b)), int(radius), 0, 360, (0, 255,0 ), 2)
                cv2.putText(img, f"{0.1}r", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                sum_jiao+=1
    sum_yuan=sum_jiao//10
    sum_jiao=sum_jiao%10
    
    cv2.putText(img, f"sum:{sum_yuan}.{sum_jiao}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    
    
    
    imshow('img', img)
    return img


def find_five_mao(img,ellipses):
    index=[]
    for i_ellipse in range(ellipses.shape[0]):
        ellipse=ellipses[i_ellipse,0,:]
        x,y,a,b,radius,score=ellipse
        
        # get a roi on ellipse center with 0.5*a and 0.5*b
        roi=img[int(y-b/2):int(y+b/2),int(x-a/2):int(x+a/2)]
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        
        lowerYellowHSV = np.array([11,43,46])
        upperYellowHSV = np.array([34,255,255])
        mask=cv2.inRange(hsv,lowerb=lowerYellowHSV,upperb=upperYellowHSV)/255
        
        # get the ratio of yellow pixel
        yellow_ratio=np.sum(mask)/(roi.shape[0]*roi.shape[1])
        if yellow_ratio>0.3:
            index.append(i_ellipse)
    return index

        
    

def test():
    # get all img unfer data/test_data
    # for each img, call det_coin
    all_files = glob.glob('data/test_data/**/*')
    is_run = True
    index=0
    while is_run:
        file=all_files[index]
        while True:
            img = cv2.imread(file)
            ret=det_coin(img)
            key=        cv2.waitKey(1)
            if key == 27:
                is_run = False
                break
            # char n
            elif key == 110:
                index+=1
                break
            # char s
            elif key == 115:
                basename=os.path.basename(file)
                cv2.imwrite(f'output/{basename}_result.jpg', ret)
                break
        
        
        
if __name__ == '__main__':
    test()
    cv2.destroyAllWindows()