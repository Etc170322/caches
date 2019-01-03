#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo'

import os
import time

'''
#run a program again and again 
#try run a program 
#parameter：
    strCmd      program cmd line
    intTimes    run how many times, "-1" means not stop,default is -1,
    intDelay    delay seconds
#return：
#
'''
def loopurn (strCmd, intTimes = -1, intDelay = 15):
    try:
        while intTimes:
            if intTimes>0:
                print("[remain %d times] " %(intTimes),end = "")
                intTimes -= 1
            print ("after %d seconds to run program:[%s] " % (intDelay , strCmd) )
            time.sleep(intDelay)
            os.system(strCmd) 
            
    except Exception as e :
        print (e)
        pass        


if __name__ == '__main__':
    loopurn ("python Info_Ext_main.py --mode=demo")
    
