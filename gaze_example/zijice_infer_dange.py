import os
import ctypes
import argparse
import sys


def shixian_det(pic_list):
    data = ""
    for i in range(len(pic_list)):
        data += pic_list[i]+","
    
    data = data[:-1]
    
    so_path = "./libsmart_classroom_demo.so"
    
    # path1 = r"C:\Users\49570\Desktop\python_dll\dll_path"
    ll = ctypes.cdll.LoadLibrary
    
    # lib = ll("/home/jzz/openvino_smart/libsmart_classroom_demo.so")
    lib = ll(so_path)
    
    main1 = lib.recognize_smart_classroom
    #main1.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
    
    main1.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
    main1.restype=ctypes.c_char_p
    
    STR_face=bytes(data,'utf-8')
    STR_res=bytes("",'utf-8')
    # ctypes.cast(STR, ctypes.POINTER(ctypes.c_char))
    # ctypes.cast(str)
    str_return = main1(STR_face,STR_res,1)
    str_get = str(str_return, encoding = "utf-8")
    str_get = str_get.split(",")[:-1]
    
    res_list = []
    for i0 in str_get:
        res_list.append(int(i0))
    
    return res_list
    



imglist=["/home/jzz/gaze_example/pic_list/1.png",
        "/home/jzz/gaze_example/pic_list/2.png",
        "/home/jzz/gaze_example/pic_list/3.png",
        "/home/jzz/gaze_example/pic_list/1.png",
        "/home/jzz/gaze_example/pic_list/2.png",
        "/home/jzz/gaze_example/pic_list/3.png"]

sx_flag = shixian_det(imglist)
print(sx_flag) #  [1, 1, 1, 1, 1, 1]




