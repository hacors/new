#pragma once
#include <iostream>
#include <fstream>
#include <queue>
using namespace std;
const int priorCom = 3;

enum class type
{
	RA,
	PA,
	APA,
	RPA,
	RAPA,
	LP,
	RLP,
	SA,
	RSA
};
class route;
class movingList;
class station
{
public:
	int _delayNum, _curLoad;			//�ֱ�Ϊ��ǰվ�������������ǰվ�ڳ˿���
	int _maxDeliver, _curDeliver; //�ֱ��ǵ������ɷ����˿������ѷ����˿���
	int _neighborNum;
	int _stationId;
	movingList *_stationPassenger;

	station(); //����ʱȫȡ��ֵ
	~station();
	void setMaxDeliver();				//�����ھ����趨վ������
	void clear();								//���ĳһվ����������
	void add(int refe);					//��������
	void remove(int refe);			//�����վ��ͨ��ĳһ�ھӵ�����
	bool exitRoute(int refe);		//�ж�һ�����Ƿ����
	route *findRoute(int refe); //����ĳһ�ߣ�û���򷵻ؿ�ָ��
	route *getHead() { return _head; };
	route *getRear() { return _rear; };
	friend ostream &operator<<(ostream &out, const station &temp);

private:
	route *_head, *_rear;
};
class route
{
public:
	route *_pre, *_next;
	int _curLoad, _maxLoad;
	int _dirDistance; //ֱ�Ӿ��룬Ĭ��Ϊ1
	bool _break, _visited;

	route(int refe = -1);										//���ñ�id��Ĭ��Ϊ-1
	void setMaxLoad(int refe1, int refe2);	//�����ٽ�վ���趨���������ɣ�
	void setDistance(int refe1, int refe2); //�����ٽ�վ���趨���루���ɣ�
	friend ostream &operator<<(ostream &out, const route *temp);
	int _stationId; //��id
};
class breakRoute
{ //��ָ�����·�ṹ
public:
	int prior; //�ָ�������Ȩ
	breakRoute(int refe1 = -1, int refe2 = -1)
	{
		prior = 0;
		station1 = refe1;
		station2 = refe2;
	};
	int getStation1() { return station1; };
	int getStation2() { return station2; };
	friend bool operator<(breakRoute refe1, breakRoute refe2);
	friend bool operator==(breakRoute refe1, breakRoute refe2);
	friend ostream &operator<<(ostream &out, const breakRoute &temp);

private:
	int station1, station2;
};
class distanceinfo
{ //·����Ϣ
public:
	int _distance; //���·��
	int _nextChoose;
	int _linkOf_1, _linkOf_2, _linkOf_3; //�����������֮�����·�ĸ���
	int _commonNei;											 //�����������֮�乲ͬ�ھӵĸ���

	distanceinfo();
	void initial(int refe1, int refe2);
	int getTarget() { return _target; };
	int getDirDis() { return _dirDis; };
	void setDirDis(int refe) { _dirDis = refe; };
	void clear(bool deep); //��ʼ�������ѡ��
	friend bool operator<(distanceinfo refe1, distanceinfo refe2);

private:
	int _dirDis;				//ֱ�Ӿ���
	int _from, _target; //վ��id
};
class moving
{
public:
	moving *_next, *_pre;
	int _movingFrom, _movingTo; //��ǰ�˶�·��
	int _distance, _haveGone;
	bool _onPlane; //�Ƿ���·;��
	bool _finish;	//�Ƿ����
	bool _new;		 //�Ƿ�����һ�ָö�������
	bool _wait;

	moving(int refe1 = -1, int refe2 = -1);
	moving(moving *ptr); //���ƹ��캯��
	int getTripFrom() { return _tripFrom; };
	int getTripTo() { return _tripTo; };
	friend ostream &operator<<(ostream &out, const moving *temp);

private:
	int _tripFrom, _tripTo; //�����˶�·��
};
class movingList
{ //����
public:
	int _passengerNum; //��ǰ������

	movingList();																 //��Ҫ��ͷβ�ս��
	~movingList();															 //��յĻ����������β���
	void newMoving(int refe1, int refe2);				 //���һ���˶����ӣ���������ͷ��
	void acceptMoving(moving *ptr, bool finish); //β������һ������
	void deleteMoving(moving *ptr);							 //ȥ��ĳһ��������
	void print();
	void clear(); //�������
	moving *getHead() { return _head; };
	moving *getRear() { retur	n _rear; };

private:
	moving *_head, *_rear; //ͷβ���
};
class myGraph
{
public:
	int _rowId, _colId; //��������
	int _congestedStation;
	int _congestedRoute;
	float _congestedRate;
	long _allSum; //�������еĳ˿���
	route *_tempRoute;
	moving *_tempMoving, *_tempDelete;
	priority_queue<breakRoute> _recoveryList;

	myGraph(int size = 0, float rate = 0.5); //�����ڽӱ�ͼ��Ĭ�ϴ�СΪ0���������ɱ���Ϊ0.5
	~myGraph();															 //ע��������б߼�β���
	void establishByInput();								 //���뽨��ͼ
	void establishByBA(int linkNum);				 //BA����ͼ
	void estavlishByRate(float rate);				 //�������ͼ
	void addRoute(int refe1, int refe2);		 //��������
	void addRoute();												 //�����������
	void updateLoad();											 //����ͼ�����˽ṹ����ͼ��������
	void updateDistance();									 //����������·�ľ���
	void updatePriorLP();										 //������·�����ȼ���Ϣ
	void updatePriorNew();									 //���¹�ͬ�ھ���
	bool check();														 //���ͼ����ͨ��
	void printGraph();											 //��ӡͼ

	void graphBreak(float rate);
	void setPriority(type choose);
	void graphRecovery();
	int getPriority(int refe1, int refe2, type choose);
	bool exitRoute(int refe1, int refe2); //�ж�ĳһ�����Ƿ����
	void printBreakList();

	void initialDistance();				 //����ͼ�ͱ��Ƿ��ƻ��ó��������
	void clearDistance(bool deep); //�Ƿ��������Ϣ
	void findNextFloyd();					 //��floyd�㷨��ȡÿһ��վ�����̾��뼰�ƶ���
	void findNextDij();						 //��dijkstras�㷨
	void printDistance();					 //�����������
	void printAnalysis();					 //���������ȵĹ�ϵ

	void printRoute(int refe1, int refe2); //��ӡ���·��
	void addLoad(float rate);							 //������ͼ����һ���ĺ��أ�������Ŀ
	void takeMove(movingList *tempPtr);		 //��ĳ��vվ�������˶�һ�Σ����º���״̬
	void allTakeMove();										 //����ͼ�˶�һ��
	void clearLoad();											 //��պ���
	void printMovingList();								 //�������ͼƬ������
	int getMovingNum();
	double congestedRate(); //�鿴����ָ��
private:
	int _stationNum;		//ͼվ������
	int _routeNum;			//��ǰ����ı�����
	float _produceRate; //ͼ���������ӵı���
	queue<breakRoute> _breakList;
	station *_graph; //���������
	distanceinfo **_moveChoice;
};