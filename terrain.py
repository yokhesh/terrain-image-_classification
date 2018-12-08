import cv2;
import numpy as np;
from matplotlib import pyplot as plt

#Load Images
img_lf = cv2.imread('Grass.JPG',0);
#img_lf = cv2.cvtColor(img_lf,cv2.COLOR_BGR2GRAY);

img_lf1 = cv2.imread('Grass1.JPG',0);
#img_lf = cv2.cvtColor(img_lf,cv2.COLOR_BGR2GRAY);

img_st = cv2.imread('interior.JPG',0);
#img_st = cv2.cvtColor(img_st,cv2.COLOR_BGR2GRAY);

img_st1 = cv2.imread('interior1.JPG',0);
#img_st = cv2.cvtColor(img_st,cv2.COLOR_BGR2GRAY);

img_lfi = cv2.imread('pathway.JPG',0);
#img_lfi = cv2.cvtColor(img_lfi,cv2.COLOR_BGR2GRAY);

img_lfi1 = cv2.imread('pathway1.JPG',0);
#img_lfi = cv2.cvtColor(img_lfi,cv2.COLOR_BGR2GRAY);

img_rg = cv2.imread('road.JPG',0);
#img_rg = cv2.cvtColor(img_rg,cv2.COLOR_BGR2GRAY);

img_rgi = cv2.imread('staircase.JPG',0);
#img_rgi = cv2.cvtColor(img_rgi,cv2.COLOR_BGR2GRAY);

img_rgi1 = cv2.imread('staircase1.JPG',0);
#img_rgi = cv2.cvtColor(img_rgi,cv2.COLOR_BGR2GRAY);


#img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY);
#SURF extraction
surf = cv2.SURF();
kp_lf,des_lf = surf.detectAndCompute(img_lf,None);
kp_lf1,des_lf1 = surf.detectAndCompute(img_lf1,None);
kp_st,des_st = surf.detectAndCompute(img_st,None);
kp_st1,des_st1 = surf.detectAndCompute(img_st1,None);
kp_lfi,des_lfi = surf.detectAndCompute(img_lfi,None);
kp_lfi1,des_lfi1 = surf.detectAndCompute(img_lfi1,None);
kp_rg,des_rg = surf.detectAndCompute(img_rg,None);
kp_rgi,des_rgi = surf.detectAndCompute(img_rgi,None);
kp_rgi1,des_rgi1 = surf.detectAndCompute(img_rgi1,None);
# Setting up samples and responses for kNN
samples_lf = np.array(des_lf);
responses_lf = np.arange(len(kp_lf),dtype = np.float32);

samples_lf1 = np.array(des_lf1);
responses_lf1 = np.arange(len(kp_lf1),dtype = np.float32);

samples_st = np.array(des_st);
responses_st = np.arange(len(kp_st),dtype = np.float32);

samples_st1 = np.array(des_st1);
responses_st1 = np.arange(len(kp_st1),dtype = np.float32);

samples_lfi = np.array(des_lfi);
responses_lfi = np.arange(len(kp_lfi),dtype = np.float32);

samples_lfi1 = np.array(des_lfi1);
responses_lfi1 = np.arange(len(kp_lfi1),dtype = np.float32);

samples_rg = np.array(des_rg);
responses_rg = np.arange(len(kp_rg),dtype = np.float32);

samples_rgi = np.array(des_rgi);
responses_rgi = np.arange(len(kp_rgi),dtype = np.float32);

samples_rgi1 = np.array(des_rgi1);
responses_rgi1 = np.arange(len(kp_rgi1),dtype = np.float32);


# kNN training
knn_lf = cv2.KNearest();
knn_lf.train(samples_lf,responses_lf);

knn_lf1 = cv2.KNearest();
knn_lf1.train(samples_lf1,responses_lf1);

knn_st = cv2.KNearest();
knn_st.train(samples_st,responses_st);

knn_st1 = cv2.KNearest();
knn_st1.train(samples_st1,responses_st1);

knn_lfi = cv2.KNearest();
knn_lfi.train(samples_lfi,responses_lfi);

knn_lfi1 = cv2.KNearest();
knn_lfi1.train(samples_lfi1,responses_lfi1);

knn_rg = cv2.KNearest();
knn_rg.train(samples_rg,responses_rg);

knn_rgi = cv2.KNearest();
knn_rgi.train(samples_rgi,responses_rgi);

knn_rgi1 = cv2.KNearest();
knn_rgi1.train(samples_rgi1,responses_rgi1);
#testing
#test_list = ['test.JPG','test1.JPG','test2.JPG','test3.JPG','test4.JPG','test5.JPG','test6.JPG','test7.JPG','test8.JPG','test9.JPG','test10.JPG','test11.JPG','test12.JPG','test13.JPG','test14.JPG','test15.JPG','test16.JPG','test17.JPG','test18.JPG','test19.JPG','test20.JPG','test21.JPG','test22.JPG','test23.JPG','test24.JPG','test25.JPG','test26.JPG','test27.JPG','test28.JPG','test29.JPG','test30.JPG','test31.JPG','test32.JPG','test33.JPG','test34.JPG','test35.JPG','test36.JPG','test37.JPG','test38.JPG','test39.JPG','test40.JPG','test41.JPG','test42.JPG','test43.JPG','test44.JPG','test45.JPG'];
test_list = ['test.JPG','test1.JPG','test2.JPG','test3.JPG','test4.JPG','test5.JPG','test6.JPG','test7.JPG','test8.JPG','test9.JPG','test10.JPG','test11.JPG','test12.JPG','test13.JPG','test14.JPG','test15.JPG','test16.JPG','test17.JPG','test18.JPG','test19.JPG','test20.JPG','test21.JPG','test22.JPG','test23.JPG','test24.JPG','test25.JPG','test26.JPG','test27.JPG','test28.JPG','test29.JPG','test30.JPG','test31.JPG','test32.JPG','test33.JPG','test34.JPG','test35.JPG','test36.JPG','test37.JPG','test38.JPG','test39.JPG','test40.JPG'];
count_lf = 0;count_st = 0;count_lfi = 0;count_rg = 0;count_rgi = 0;
count_st1 = 0;count_lf1 = 0;count_lfi1 = 0;count_rgi1 = 0;
for i in range(len(test_list)):
    test = cv2.imread('test.JPG',0);
    kpt,dest = surf.detectAndCompute(test,None);

    count_lf = 0;count_st = 0;count_lfi = 0;count_rg = 0;count_rgi = 0;
    count_st1 = 0;count_lf1 = 0;count_lfi1 = 0;count_rgi1 = 0;
    count= [0,0,0,0,0,0,0,0,0];
    resu = [];
    for h,des in enumerate(dest):
        des = np.array(des,np.float32).reshape((1,64));
    
        retval_lf, results_lf, neigh_resp_lf, dists_lf = knn_lf.find_nearest(des,1);
        retval_lf1, results_lf1, neigh_resp_lf1, dists_lf1 = knn_lf1.find_nearest(des,1);
        retval_st, results_st, neigh_resp_st, dists_st = knn_st.find_nearest(des,1);
        retval_st1, results_st1, neigh_resp_st1, dists_st1 = knn_st1.find_nearest(des,1);
        retval_lfi, results_lfi, neigh_resp_lfi, dists_lfi = knn_lfi.find_nearest(des,1);
        retval_lfi1, results_lfi1, neigh_resp_lfi1, dists_lfi1 = knn_lfi1.find_nearest(des,1);
        retval_rg, results_rg, neigh_resp_rg, dists_rg = knn_rg.find_nearest(des,1);
        retval_rgi, results_rgi, neigh_resp_rgi, dists_rgi = knn_rgi.find_nearest(des,1);
        retval_rgi1, results_rgi1, neigh_resp_rgi1, dists_rgi1 = knn_rgi1.find_nearest(des,1);
    
        res_lf,dist_lf =  int(results_lf[0][0]),dists_lf[0][0];
        resu.append(dist_lf);
        res_lf1,dist_lf1 =  int(results_lf1[0][0]),dists_lf1[0][0];
        resu.append(dist_lf1);
        res_st,dist_st =  int(results_st[0][0]),dists_st[0][0];
        resu.append(dist_st);
        res_st,dist_st1 =  int(results_st1[0][0]),dists_st1[0][0];
        resu.append(dist_st1);
        res_lfi,dist_lfi =  int(results_lfi[0][0]),dists_lfi[0][0];
        resu.append(dist_lfi);
        res_lfi,dist_lfi1 =  int(results_lfi1[0][0]),dists_lfi1[0][0];
        resu.append(dist_lfi1);
        res_rg,dist_rg =  int(results_rg[0][0]),dists_rg[0][0];
        resu.append(dist_rg);
        res_rgi,dist_rgi =  int(results_rgi[0][0]),dists_rgi[0][0];
        resu.append(dist_rgi);
        res_rgi1,dist_rgi1 =  int(results_rgi1[0][0]),dists_rgi1[0][0];
        resu.append(dist_rgi1);



        minIndex = resu.index(min(resu));
        if (minIndex == 0):
            count_lf = count_lf+1;
            count[0] = count[0]+1;
        if (minIndex == 1):
            count_lf1 = count_lf1+1;
            count[1] = count[1]+1;
        if (minIndex == 2):
            count_st = count_st+1;
            count[2] = count[2]+1;
        if (minIndex == 3):
            count_st1 = count_st1+1;
            count[3] = count[3]+1;
        if (minIndex == 4):
            count_lfi = count_lfi+1;
            count[4] = count[4]+1;
        if (minIndex == 5):
            count_lfi = count_lfi+1;
            count[5] = count[5]+1;
        if (minIndex == 6):
            count_rg = count_rg+1;
            count[6] = count[6]+1;
        if (minIndex == 7):
            count_rgi = count_rgi+1;
            count[7] = count[7]+1;
        if (minIndex == 8):
            count_rgi = count_rgi+1;
            count[8] = count[8]+1;
        resu = [];
    max_in = count.index(max(count));
    if (max_in == 0):
        print('GRASS');
    if (max_in == 1):
        print('GRASS');
    if (max_in == 2):
        print('INTERIOR');
    if (max_in == 3):
        print('INTERIOR');
    if (max_in == 4):
        print('PATHWAY');
    if (max_in == 5):
        print('PATHWAY');
    if (max_in == 6):
        print('ROAD');
    if (max_in == 7):
        print('STAIRCASE');
    if (max_in == 8):
        print('STAIRCASE');
    #if dist<0.001:
     #   count = count+1;
    
#for h,des in enumerate(dest):
#    des = np.array(des,np.float32).reshape((1,64));
#    retval1, results1, neigh_resp1, dists1 = knn1.find_nearest(des,1);
#    res1,dist1 =  int(results1[0][0]),dists1[0][0];
#    if dist1<0.01:
#        count1 = count1+1;


#orb = cv2.ORB()
#kp1, des1 = orb.detectAndCompute(img1,None)
#surf = cv2.SURF(400);
#kp,des = surf.detectAndCompute(img,None);
#print (len(kp));
