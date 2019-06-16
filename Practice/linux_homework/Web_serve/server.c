/* 
 * File:   Server.c
 * Author: root
 *
 * Created on 2012年6月20日, 下午1:29
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>
 
#define SERVERIP "192.168.0.23"
#define SERVERPORT 12345
#define MAXBUFFER 256
 
int main(int argc, char** argv)
{
    int serverFd, connfd,ret;
    socklen_t len;
    struct sockaddr_in serveraddr,clientaddr;
    char readBuf[MAXBUFFER]={0};
    char ip[40]={0};
    serverFd=socket(AF_INET,SOCK_STREAM,0);//创建socket
    if(serverFd < 0)
    {
        printf("socket error:%s\n",strerror(errno));
        exit(-1);
    }
    bzero(&serveraddr,sizeof(serveraddr));
    serveraddr.sin_family=AF_INET;
    serveraddr.sin_port=htons(SERVERPORT);
    inet_pton(AF_INET,SERVERIP,&serveraddr.sin_addr);//将c语言字节序转换为网络字节序
    ret=bind(serverFd,(struct sockaddr *)&serveraddr,sizeof(serveraddr));//绑定IP和端口
    if(ret!=0)
    {
        fclose(serverFd);
        printf("bind error:%s\n",strerror(errno));
        exit(-1);
    }
    ret=listen(serverFd,5);//监听
    if(ret!=0)
    {
       fclose(serverFd);
       printf("listen error:%s\n",strerror(errno));
       exit(-1);
    }
    len=sizeof(clientaddr);
    bzero(&clientaddr,sizeof(clientaddr));
    while (1)
    {
        connfd = accept(serverFd, (struct sockaddr *) &clientaddr, &len);//接受客户端的连接
        printf("%s 连接到服务器 \n",inet_ntop(AF_INET,&clientaddr.sin_addr,ip,sizeof(ip)));
        if (serverFd < 0)
        {
            printf("accept error : %s\n", strerror(errno));
            continue;
        }
        while((ret=fread(connfd,readBuf,MAXBUFFER)))//读客户端发送的数据
        {
            fwrite(connfd,readBuf,MAXBUFFER);//写回客户端
            bzero(readBuf,MAXBUFFER);
        }
        if(ret==0)
        {
            printf("客户端关闭连接\n");         
        }else
        {
            printf("read error:%s\n",strerror(errno));
        }
        fclose(connfd);
    }
    fclose(serverFd);
    return 0;
}