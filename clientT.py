#!/usr/bin/python
#*-* coding: utf-8 *-*
from socket import *

HOST='172.18.139.50'

c = socket(AF_INET, SOCK_STREAM)
print ('connecting....')
c.connect((HOST,8000))
print ('ok')
while 1:
        data = input()
        if data:
                c.send(str(data))
        else:
                continue
        print ('recive_data : ',c.recv(1024))
c.close()