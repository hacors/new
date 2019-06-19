#include <iostream>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using namespace std;
void main()
{
    //初始化winsock
    WSADATA wsDARA;                       //用于处理socket的数据结构
    WORD Ver = MAKEWORD(2, 2);            //声明调用不同的Winsock版本
    int wsOK = WSAStartup(Ver, &WSADATA); //启动命令
    if (wsOK != 0)
    {
        cerr << "无法初始化winsock" << endl; //一般cerr用于输出错误信息，不需要缓存
        return;
    }

    //建立一个socket
    SOCKET listening = socket(AF_INET, SOCK_STREAM, 0); //协议族为domain、协议类型为type
    if (listening == INVALID_SOCKET)
    {
        cerr << "无法建立socket" << endl;
        return;
    }

    //将socket连接到一个固定的ip和port
    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(54000);
    hint.sin_addr.S_un.S_addr = INADDR_ANY;
    bind(listening, (sockaddr *)&hint, sizeof(hint));

    //socket开始监听
    listen(listening, SOMAXCONN);

    //等待连接
    sockaddr_in client;
    int clientsize = sizeof(client);
    SOCKET clientSocket = accept(listening, (sockaddr *)&client, &clientsize);
    char host[NI_MAXHOST];    //客户端名
    char service[NI_MAXHOST]; //服务端名
    ZeroMemory(host, NI_MAXHOST);
    ZeroMemory(service, NI_MAXHOST);

    //监听结束

    //处理数据，接收并返回消息

    //关闭socket
    //关闭winsock服务
}