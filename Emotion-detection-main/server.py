from socket import *

serverSocket=socket(AF_INET,SOCK_STREAM)
serverSocket.bind(("10.209.92.109",1200))
#serverSocket.bind(("192.168.1.12",1200))

serverSocket.listen(20)
print("the server is ready to accept info...")
connectionSocket,address=serverSocket.accept()
print("client 1 online")
connectionSocket2,address2=serverSocket.accept()
print("client 2 online")
while True:
    message=connectionSocket.recv(1024).decode()
    print("got the message from the client: "+message)
    if message and connectionSocket2:
        connectionSocket2.send((message).encode())
        #sconnectionSocket.send((message+"haha").encode())
    elif not connectionSocket2:
        break
connectionSocket.close()