from socket import *
import threading
import time

serverSocket=socket(AF_INET,SOCK_STREAM)
serverSocket.bind(("10.209.92.109",1200))
#serverSocket.bind(("192.168.142.132",1200))
#serverSocket.bind(("192.168.1.12",1200))
#serverSocket.bind(("127.0.0.1",1200))

serverSocket.listen(20)
print("the server is ready to accept info...")
connectionSocket,address=serverSocket.accept()
print("client 1 online")
connectionSocket2,address2=serverSocket.accept()
print("client 2 online")
message=''


def message2music():
    while True:
        global message
        if connectionSocket2:
            if  message!="":
                connectionSocket2.send((message).encode())
                print("message to the music generation client",message)
                message=""
                time.sleep(3.7)
                #sconnectionSocket.send((message+"haha").encode())
            else:
                connectionSocket2.send(("neutral").encode())
                print("message to the music generation client neutral")
                time.sleep(3.7)
        else:
            connectionSocket.close()
            connectionSocket2.close()
            break



def listening2emo():
    while True: 
        global message
        message=connectionSocket.recv(1024).decode()
        print("got the message from the client: "+message)


        

def main():
    thread1=threading.Thread(target=listening2emo,name="T1")
    thread2=threading.Thread(target=message2music,name="T2")
    thread1.start()
    thread2.start()
    thread1.join()
    print("all done")

if __name__=='__main__':
    main()