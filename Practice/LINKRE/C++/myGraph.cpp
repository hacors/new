#pragma once
#include "myGraph.h"
#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <queue>
#include <string>
using namespace std;

station::station()
{
	_delayNum = _maxDeliver = _curDeliver = _curLoad = 0;
	_neighborNum = 0;
	_stationId = -1;
	_head = new route();
	_rear = new route();
	_stationPassenger = new movingList();
	_head->_next = _rear;
	_rear->_pre = _head;
}
station::~station()
{
	clear();
	delete _head;
	delete _rear;
	delete _stationPassenger;
}
void station::setMaxDeliver()
{
	_maxDeliver = _neighborNum; //存疑
}
void station::clear()
{
	route *tempPtr = _head->_next;
	route *deletePtr;
	while (tempPtr != _rear)
	{
		deletePtr = tempPtr;
		tempPtr = tempPtr->_next;
		delete deletePtr;
	}
}
void station::add(int refe)
{
	route *tempPtr = new route(refe);
	tempPtr->_next = _head->_next;
	tempPtr->_pre = _head;
	_head->_next->_pre = tempPtr;
	_head->_next = tempPtr;
	_neighborNum++;
}
void station::remove(int refe)
{
	route *tempPtr = _head->_next;
	route *deletePtr;
	bool done = false;
	while (tempPtr != _rear && done == false)
	{
		deletePtr = tempPtr;
		tempPtr = tempPtr->_next;
		if (deletePtr->_stationId == refe)
		{
			deletePtr->_pre->_next = deletePtr->_next;
			deletePtr->_next->_pre = deletePtr->_pre;
			delete deletePtr;
			_neighborNum--;
			done = true;
		}
	}
}
route *station::findRoute(int refe)
{
	route *tempPtr = _head->_next;
	while (tempPtr != _rear)
	{
		if (tempPtr->_stationId == refe)
			return tempPtr;
		tempPtr = tempPtr->_next;
	}
	return nullptr;
}
bool station::exitRoute(int refe)
{
	route *tempPtr = _head->_next;
	while (tempPtr != _rear)
	{
		if (tempPtr->_stationId == refe)
			return true;
		tempPtr = tempPtr->_next;
	}
	return false;
}
ostream &operator<<(ostream &out, const station &temp)
{
	out << "站点" << temp._stationId << " 延误[" << temp._delayNum << "/" << temp._curLoad << "]出发["
		<< temp._curDeliver << "/" << temp._maxDeliver << "]"
		<< " 边数" << temp._neighborNum << "：";
	return out;
}
route::route(int refe)
{
	_stationId = refe;
	_pre = _next = nullptr;
	_curLoad = _maxLoad = 0;
	_dirDistance = 1;
	_break = false;
	_visited = false;
}
void route::setMaxLoad(int refe1, int refe2)
{ //存疑
	_maxLoad = 10000;
}
void route::setDistance(int refe1, int refe2)
{ //存疑
	_dirDistance = 1;
}
ostream &operator<<(ostream &out, route *temp)
{
	out << " " << temp->_stationId << " [" << temp->_curLoad << "/" << temp->_maxLoad << "]"
		<< "(" << temp->_dirDistance << ")->";
	return out;
}
bool operator<(breakRoute refe1, breakRoute refe2)
{
	return refe1.prior < refe2.prior;
}
bool operator==(breakRoute refe1, breakRoute refe2)
{
	return (refe1.station1 == refe2.station1 && refe1.station2 == refe2.station2) ? true : false;
}
ostream &operator<<(ostream &out, const breakRoute &temp)
{
	out << " [" << temp.station1 << "," << temp.station2 << "](" << temp.prior << ")->";
	return out;
}
distanceinfo::distanceinfo()
{
	_from = -1;
	_target = -1;
	_dirDis = -1;
	_distance = -1;
	_nextChoose = -1;
	_commonNei = 0;
}
void distanceinfo::initial(int refe1, int refe2)
{
	_from = refe1;
	_target = refe2;
	_dirDis = (_from == _target) ? 0 : -1;
	_nextChoose = _target;
	_distance = -1;
	_distance = _dirDis;
};
void distanceinfo::clear(bool deep)
{
	_distance = deep ? ((_from == _target) ? 0 : -1) : _dirDis;
	_nextChoose = _target;
}
bool operator<(distanceinfo refe1, distanceinfo refe2)
{
	if (refe1._distance == refe2._distance)
		return refe1.getTarget() > refe2.getTarget();
	else
		return refe1._distance > refe2._distance;
}
moving::moving(int refe1, int refe2)
{
	_tripFrom = _movingFrom = _movingTo = refe1;
	_tripTo = refe2;
	_distance = _haveGone = 0;
	_next = _pre = nullptr;
	_onPlane = _finish = false;
	_new = false; //本轮行动
	_wait = false;
}
moving::moving(moving *ptr)
{
	_movingFrom = _movingTo = ptr->_movingTo;
	_distance = _haveGone = 0;
	_onPlane = _finish = false;
	_new = true; //下轮行动
	_wait = false;
	_tripFrom = ptr->_tripFrom;
	_tripTo = ptr->_tripTo;
}
ostream &operator<<(ostream &out, const moving *temp)
{
	out << temp->_tripFrom << "/" << temp->_tripTo << "(" << temp->_movingFrom << "/"
		<< temp->_movingTo << ")[" << temp->_haveGone << "/" << temp->_distance << "]->";
	return out;
}
movingList::movingList()
{
	_head = new moving();
	_rear = new moving();
	_head->_next = _rear;
	_rear->_pre = _head;
	_passengerNum = 0;
}
movingList::~movingList()
{
	clear();
	delete _head;
	delete _rear;
}
void movingList::newMoving(int refe1, int refe2)
{
	moving *tempPtr = new moving(refe1, refe2);
	tempPtr->_next = _rear;
	tempPtr->_pre = _rear->_pre;
	_rear->_pre->_next = tempPtr;
	_rear->_pre = tempPtr;
	_passengerNum++;
}
void movingList::acceptMoving(moving *ptr, bool finish)
{
	moving *tempPtr = new moving(ptr);
	tempPtr->_finish = finish;
	tempPtr->_next = _rear;
	tempPtr->_pre = _rear->_pre;
	_rear->_pre->_next = tempPtr;
	_rear->_pre = tempPtr;
	_passengerNum++;
}
void movingList::deleteMoving(moving *tempPtr)
{
	tempPtr->_pre->_next = tempPtr->_next;
	tempPtr->_next->_pre = tempPtr->_pre;
	delete tempPtr;
	_passengerNum--;
}
void movingList::print()
{
	moving *tempPtr = _head->_next;
	cout << "乘客数(" << _passengerNum << ") ";
	while (tempPtr != _rear)
	{
		cout << tempPtr;
		tempPtr = tempPtr->_next;
	}
}
void movingList::clear()
{
	moving *tempPtr = _head->_next;
	moving *deletePtr;
	while (tempPtr != _rear)
	{
		deletePtr = tempPtr;
		tempPtr = tempPtr->_next;
		deleteMoving(deletePtr);
	}
}
myGraph::myGraph(int size, float rate)
{
	_rowId = _colId = 0;
	_congestedRoute = _congestedStation = 0;
	_tempRoute = nullptr;
	_stationNum = size;
	_routeNum = 0;
	_produceRate = rate;
	_allSum = 0;
	_graph = new station[_stationNum];
	_moveChoice = new distanceinfo *[_stationNum];
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_graph[_rowId]._stationId = _rowId;
		_moveChoice[_rowId] = new distanceinfo[_stationNum];
	}
}
myGraph::~myGraph()
{
	delete[] _graph;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
		delete[] _moveChoice[_rowId];
	delete[] _moveChoice;
}
void myGraph::establishByInput()
{
	ifstream read;
	read.open("d:\\text.txt");
	int id = 0;
	int num1, num2, num3;
	while (id < _stationNum)
	{ //因为txt从1开始计数
		read >> num1 >> num2 >> num3;
		addRoute(num1 - 1, num2 - 1);
		_graph[num1 - 1].findRoute(num2 - 1)->_dirDistance = num3;
		_graph[num2 - 1].findRoute(num1 - 1)->_dirDistance = num3;
		id = num1;
	}
	read.close();
}
void myGraph::establishByBA(int BAlinkNum)
{
	int rounds, pos, count, cursor, temp;
	bool success;
	int *store = new int[BAlinkNum];
	for (_rowId = 0; _rowId < BAlinkNum; _rowId++)
	{ //初始化
		for (_colId = _rowId + 1; _colId < BAlinkNum; _colId++)
		{
			addRoute(_rowId, _colId);
		}
	}
	for (_rowId = BAlinkNum; _rowId < _stationNum; _rowId++)
	{
		for (rounds = 0; rounds < BAlinkNum; rounds++)
		{
			success = false;
			while (!success)
			{
				success = true;
				cursor = 0;
				count = _graph[cursor]._neighborNum;
				pos = rand() % (_routeNum * 2); //控制落点区间
				while (count <= pos)
				{
					count = count + _graph[cursor]._neighborNum;
					cursor++;
				}
				temp = rounds;
				while (temp--)
				{
					if (cursor == store[temp])
						success = false;
				}
			}
			store[rounds] = cursor;
		}
		for (rounds = 0; rounds < BAlinkNum; rounds++)
		{
			addRoute(_rowId, store[rounds]);
		}
	}
	delete[] store;
}
void myGraph::estavlishByRate(float rate)
{
	int generate;
	generate = static_cast<int>(rate * _stationNum);
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			if (rand() % _stationNum <= generate)
			{
				_graph[_rowId].add(_colId);
			}
		}
	}
}
void myGraph::addRoute(int refe1, int refe2)
{
	if (_graph[refe1].exitRoute(refe2) || _graph[refe2].exitRoute(refe1) || refe1 == refe2)
		return;
	_graph[refe1].add(refe2);
	_graph[refe2].add(refe1);
	_routeNum++;
}
void myGraph::addRoute()
{
	bool check = true; //判断是否相等
	int refe1, refe2;
	if (_stationNum == 0)
		return; //防止报错
	refe1 = rand() % _stationNum;
	while (check)
	{
		refe2 = rand() % _stationNum;
		if (refe1 != refe2)
			check = false;
	}
	addRoute(refe1, refe2);
}
void myGraph::updateLoad()
{
	int tempNum1, tempNum2;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_graph[_rowId].setMaxDeliver();
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			_colId = _tempRoute->_stationId;
			tempNum1 = _graph[_rowId]._neighborNum;
			tempNum2 = _graph[_colId]._neighborNum;
			_tempRoute->setMaxLoad(tempNum1, tempNum2);
			_tempRoute = _tempRoute->_next;
		}
	}
}
void myGraph::updateDistance()
{
	int tempNum1, tempNum2;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			_colId = _tempRoute->_stationId;
			tempNum1 = _graph[_rowId]._neighborNum;
			tempNum2 = _graph[_colId]._neighborNum;
			_tempRoute->setDistance(tempNum1, tempNum2);
			_tempRoute = _tempRoute->_next;
		}
	}
}
void myGraph::updatePriorLP()
{
	int cursor;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{ //初始化
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			_moveChoice[_rowId][_colId]._linkOf_1 = (_moveChoice[_rowId][_colId].getDirDis() <= 0) ? 0 : 1;
			_moveChoice[_rowId][_colId]._linkOf_2 = 0;
			_moveChoice[_rowId][_colId]._linkOf_3 = 0;
		}
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			for (cursor = 0; cursor < _stationNum; cursor++)
				_moveChoice[_rowId][_colId]._linkOf_2 += _moveChoice[_rowId][cursor]._linkOf_1 * _moveChoice[cursor][_colId]._linkOf_1;
		}
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			for (cursor = 0; cursor < _stationNum; cursor++)
				_moveChoice[_rowId][_colId]._linkOf_3 += _moveChoice[_rowId][cursor]._linkOf_2 * _moveChoice[cursor][_colId]._linkOf_2;
		}
	}
}
void myGraph::updatePriorNew()
{
	int **compare = new int *[_stationNum];
	bool *visited = new bool[_stationNum];
	queue<int> searchQueue;
	route *tempPtr;
	int temp, pos, cursor;
	int deep, stack; //记录深度以及每一层最后一点
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{ //初始化
		compare[_rowId] = new int[_stationNum];
		visited[_rowId] = false;
		for (_colId = 0; _colId < _stationNum; _colId++)
			compare[_rowId][_colId] = 0;
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		deep = 1;
		while (!searchQueue.empty())
			searchQueue.pop(); //清空
		for (pos = 0; pos < _stationNum; pos++)
			visited[pos] = false;
		searchQueue.push(_rowId);
		visited[_rowId] = true;
		stack = _graph[_rowId].getRear()->_pre->_stationId;
		while (!searchQueue.empty() && deep < priorCom)
		{
			temp = searchQueue.front();
			tempPtr = _graph[temp].getHead()->_next;
			while (tempPtr != _graph[temp].getRear())
			{
				if (!visited[tempPtr->_stationId] && tempPtr->_break == false)
				{
					visited[tempPtr->_stationId] = true;
					compare[_rowId][tempPtr->_stationId] = 1;
					searchQueue.push(tempPtr->_stationId);
				}
				tempPtr = tempPtr->_next;
			}
			if (searchQueue.front() == stack)
			{
				stack = searchQueue.back();
				deep++;
			}
			searchQueue.pop();
		}
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			for (cursor = 0; cursor < _stationNum; cursor++)
				_moveChoice[_rowId][_colId]._commonNei += compare[_rowId][cursor] * compare[_colId][cursor];
		}
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
		delete[] compare[_rowId];
	delete[] compare;
	delete[] visited;
}
bool myGraph::check()
{
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			if (_moveChoice[_rowId][_colId]._distance == -1)
				return false;
		}
	}
	return true;
}
void myGraph::printGraph()
{
	cout << endl
		 << "输出图结构：" << endl;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		cout << _graph[_rowId];
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			if (_tempRoute->_break == false)
				cout << _tempRoute;
			_tempRoute = _tempRoute->_next;
		}
		cout << endl;
	}
	cout << "其中总边数为：" << _routeNum << " 总粒子数为：" << getMovingNum() << endl
		 << endl;
}
void myGraph::graphBreak(float rate)
{
	int generate = static_cast<int>(rate * _stationNum);
	int temp;
	while (!_breakList.empty())
		_breakList.pop();
	_breakList.push(breakRoute()); //初始化
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			_tempRoute->_visited = false;
			_tempRoute = _tempRoute->_next;
		}
	}
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			if (_tempRoute->_visited == false)
			{
				temp = _tempRoute->_stationId;
				_tempRoute->_visited = true;
				_graph[temp].findRoute(_rowId)->_visited = true;
				if (rand() % _stationNum <= generate)
				{
					_tempRoute->_break = true;
					_graph[temp].findRoute(_rowId)->_break = true; //对应边的破坏
					_graph[_rowId]._neighborNum--;
					_graph[temp]._neighborNum--;
					_moveChoice[_rowId][temp].setDirDis(-1);
					_moveChoice[temp][_rowId].setDirDis(-1);
					_routeNum--;
					_breakList.push(breakRoute(_rowId, temp));
				}
			}
			_tempRoute = _tempRoute->_next;
		}
	}
}
void myGraph::graphRecovery()
{
	breakRoute temp = _recoveryList.top();
	_recoveryList.pop();
	int refe1 = temp.getStation1();
	int refe2 = temp.getStation2();
	_graph[refe1].findRoute(refe2)->_break = false;
	_graph[refe2].findRoute(refe1)->_break = false;
	_graph[refe1]._neighborNum++;
	_graph[refe2]._neighborNum++;
	_moveChoice[refe1][refe2].setDirDis(_graph[refe1].findRoute(refe2)->_dirDistance);
	_moveChoice[refe2][refe1].setDirDis(_graph[refe2].findRoute(refe1)->_dirDistance);
	_routeNum++;
}
int myGraph::getPriority(int refe1, int refe2, type choose)
{
	switch (choose)
	{
	case type::RA:
		return rand() % _stationNum;
		break;
	case type::PA:
		return _graph[refe1]._neighborNum * _graph[refe1]._neighborNum;
		break;
	case type::APA:
		return _graph[refe1]._neighborNum * _graph[refe1]._neighborNum * (-1);
		break;
	case type::RPA:
		return _graph[refe1]._neighborNum + _graph[refe1]._neighborNum;
		break;
	case type::RAPA:
		return (_graph[refe1]._neighborNum + _graph[refe1]._neighborNum) * (-1);
		break;
	case type::LP:
		return _moveChoice[refe1][refe2]._linkOf_2 + _moveChoice[refe1][refe2]._linkOf_3 / 5;
		break;
	case type::RLP:
		return (_moveChoice[refe1][refe2]._linkOf_2 + _moveChoice[refe1][refe2]._linkOf_3 / 5) * (-1);
		break;
	case type::SA:
		return _moveChoice[refe1][refe2]._commonNei;
		break;
	case type::RSA:
		return _moveChoice[refe1][refe2]._commonNei * (-1);
		break;
	default:
		return 0;
		break;
	}
}
void myGraph::setPriority(type choose)
{
	breakRoute temp = breakRoute();
	breakRoute cursor;
	while (!_recoveryList.empty())
		_recoveryList.pop();
	_breakList.pop();
	_breakList.push(temp);
	while (!(temp == _breakList.front()))
	{
		cursor = _breakList.front();
		cursor.prior = getPriority(cursor.getStation1(), cursor.getStation2(), choose);
		_recoveryList.push(cursor);
		_breakList.pop();
		_breakList.push(cursor);
	}
}
bool myGraph::exitRoute(int refe1, int refe2)
{
	if (_graph[refe1].exitRoute(refe2) && _graph[refe2].exitRoute(refe1))
		return true;
	else
		return false;
}
void myGraph::printBreakList()
{
	breakRoute temp = breakRoute();
	breakRoute cursor;
	_breakList.pop();
	_breakList.push(temp);
	cout << "破坏的链路" << _breakList.size() - 1 << "：";
	while (!(temp == _breakList.front()))
	{
		cursor = _breakList.front();
		cout << cursor;
		_breakList.pop();
		_breakList.push(cursor);
	}
	cout << endl
		 << endl;
}
void myGraph::initialDistance()
{
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{ //初始化
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			_moveChoice[_rowId][_colId].initial(_rowId, _colId);
		}
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			_colId = _tempRoute->_stationId;
			_moveChoice[_rowId][_colId].setDirDis(_tempRoute->_dirDistance);
			_tempRoute = _tempRoute->_next;
		}
	}
}
void myGraph::clearDistance(bool deep)
{
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{ //初始化
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			_moveChoice[_rowId][_colId].clear(deep);
		}
	}
}
void myGraph::printDistance()
{
	cout << "距离为：" << endl;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		for (_colId = 0; _colId < _stationNum; _colId++)
		{
			cout << setw(4) << _moveChoice[_rowId][_colId]._distance << "("
				 << _moveChoice[_rowId][_colId]._nextChoose << ")";
		}
		cout << endl;
	}
	cout << endl;
}
void myGraph::printAnalysis()
{
	int tempNum1, tempNum2;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_tempRoute = _graph[_rowId].getHead()->_next;
		while (_tempRoute != _graph[_rowId].getRear())
		{
			_colId = _tempRoute->_stationId;
			if (_rowId > _colId)
			{
				tempNum1 = _graph[_rowId]._neighborNum;
				tempNum2 = _graph[_colId]._neighborNum;
				cout << _moveChoice[_rowId][_colId].getDirDis() << " " << tempNum1 << " " << tempNum2 << endl;
			}
			_tempRoute = _tempRoute->_next;
		}
	}
}
void myGraph::findNextFloyd()
{
	int chooseId = 0;
	int tempDistance = -1;
	for (chooseId = 0; chooseId < _stationNum; chooseId++)
	{
		for (_rowId = 0; _rowId < _stationNum; _rowId++)
		{
			for (_colId = 0; _colId < _stationNum; _colId++)
			{
				if (_moveChoice[_rowId][chooseId]._distance != -1 && _moveChoice[chooseId][_colId]._distance != -1)
				{
					tempDistance = _moveChoice[_rowId][chooseId]._distance + _moveChoice[chooseId][_colId]._distance;
					if (tempDistance < _moveChoice[_rowId][_colId]._distance || _moveChoice[_rowId][_colId]._distance == -1)
					{
						_moveChoice[_rowId][_colId]._distance = tempDistance;
						_moveChoice[_rowId][_colId]._nextChoose = _moveChoice[_rowId][chooseId]._nextChoose;
					}
				}
			}
		}
	}
}
void myGraph::findNextDij()
{
	distanceinfo tempDisInfo;
	int topId, cursorId; //分别为当前确认的站点和遍历站点
	priority_queue<distanceinfo> queue;
	//优先级队列永远维护当前结点到所有结点最短距离中最小的那个（当然只包括未确定结点）
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{ //各自进行遍历
		tempDisInfo = _moveChoice[_rowId][_rowId];
		queue.push(tempDisInfo);
		while (!queue.empty())
		{
			tempDisInfo = queue.top();
			queue.pop();
			topId = tempDisInfo.getTarget();
			_tempRoute = _graph[topId].getHead()->_next;
			while (_tempRoute != _graph[topId].getRear())
			{
				cursorId = _tempRoute->_stationId;
				if (_moveChoice[_rowId][cursorId]._distance == -1 ||
					_moveChoice[_rowId][cursorId]._distance > _moveChoice[_rowId][topId]._distance + _tempRoute->_dirDistance)
				{
					_moveChoice[_rowId][cursorId]._distance = _moveChoice[_rowId][topId]._distance + _tempRoute->_dirDistance;
					_moveChoice[_rowId][cursorId]._nextChoose = (topId == _rowId) ? cursorId : _moveChoice[_rowId][topId]._nextChoose;
					queue.push(_moveChoice[_rowId][cursorId]);
				}
				_tempRoute = _tempRoute->_next;
			}
		}
	}
}
void myGraph::printRoute(int refe1, int refe2)
{
	int tempId;
	cout << "路径为：" << refe1 << "->";
	while (_moveChoice[refe1][refe2]._nextChoose != refe2)
	{
		tempId = _moveChoice[refe1][refe2]._nextChoose;
		cout << tempId << "->";
		refe1 = tempId;
	}
	cout << refe2 << endl
		 << endl;
}
void myGraph::addLoad(float rate)
{
	if (_stationNum == 0)
		return; //防止中断
	int tempId, tempNum;
	bool check; //取另一点不同于该点
	int generate;
	tempNum = 0;
	generate = static_cast<int>(rate * _stationNum); //是否产生粒子的参照
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		if (rand() % _stationNum <= generate)
		{
			check = true;
			while (check)
			{
				tempId = rand() % _stationNum;
				if (tempId != _rowId)
					check = false;
			}
			_graph[_rowId]._stationPassenger->newMoving(_rowId, tempId);
			_graph[_rowId]._curLoad++;
			tempNum++;
		}
	}
	_allSum = _allSum + tempNum;
}
void myGraph::takeMove(movingList *tempPtr)
{
	int tempId1, tempId2;
	_tempMoving = tempPtr->getHead()->_next;
	while (_tempMoving != tempPtr->getRear())
	{ //遍历
		if (_tempMoving->_new == false)
		{ //控制轮次
			if (_tempMoving->_onPlane == true)
			{ //路上
				_tempMoving->_haveGone++;
				if (_tempMoving->_haveGone == _tempMoving->_distance)
				{																					   //是否到站
					bool finish = (_tempMoving->_movingTo == _tempMoving->getTripTo()) ? true : false; //是否结束
					tempId1 = _tempMoving->_movingFrom;
					tempId2 = _tempMoving->_movingTo;
					_graph[tempId2]._curLoad++;						//站点荷载
					_graph[tempId1].findRoute(tempId2)->_curLoad--; //链路荷载
					_graph[tempId2]._stationPassenger->acceptMoving(_tempMoving, finish);
					_tempDelete = _tempMoving;
					_tempMoving = _tempMoving->_next;
					tempPtr->deleteMoving(_tempDelete);
					continue; //删除情况需特殊处理，只需保留指针就行了
				}
			}
			else
			{ //站里
				if (_tempMoving->_finish == true)
				{ //结束的粒子清除
					_graph[_tempMoving->_movingFrom]._curLoad--;
					_tempDelete = _tempMoving;
					_tempMoving = _tempMoving->_next;
					tempPtr->deleteMoving(_tempDelete);
					continue; //删除情况需特殊处理，只需保留指针就行了
				}
				else
				{ //需要出发的粒子
					_tempMoving->_movingTo = _moveChoice[_tempMoving->_movingFrom][_tempMoving->getTripTo()]._nextChoose;
					tempId1 = _tempMoving->_movingFrom;
					tempId2 = _tempMoving->_movingTo;
					if (_moveChoice[tempId1][tempId2]._distance != -1)
					{ //判断是否为孤立点
						bool tempCheck1 = _graph[tempId1].findRoute(tempId2)->_curLoad < _graph[tempId1].findRoute(tempId2)->_maxLoad;
						bool tempCheck2 = _graph[tempId1]._curDeliver < _graph[tempId1]._maxDeliver;
						if (tempCheck1 && tempCheck2)
						{
							if (_tempMoving->_wait == true)
							{ //注意不要重复统计
								_graph[tempId1]._delayNum--;
								_tempMoving->_wait = false;
							}
							_tempMoving->_distance = _moveChoice[tempId1][tempId2]._distance;
							_graph[tempId1].findRoute(tempId2)->_curLoad++; //注意为双向计算
							_graph[tempId1]._curLoad--;
							_graph[tempId1]._curDeliver++;
							_tempMoving->_onPlane = true; //及时更改状态
						}
						else
						{
							if (_tempMoving->_wait == false)
							{ //注意不要重复统计
								_graph[tempId1]._delayNum++;
								_tempMoving->_wait = true;
							}
						}
					}
					else
					{
						if (_tempMoving->_wait == false)
						{ //注意不要重复统计
							_graph[tempId1]._delayNum++;
							_tempMoving->_wait = true;
						}
					}
				}
			}
		}
		else
			_tempMoving->_new = false;
		_tempMoving = _tempMoving->_next;
	}
}
void myGraph::allTakeMove()
{
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_graph[_rowId]._curDeliver = 0;
		takeMove(_graph[_rowId]._stationPassenger);
	}
}
void myGraph::clearLoad()
{
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		_graph[_rowId]._stationPassenger.clear();
	}
}
void myGraph::printMovingList()
{
	cout << "输出所有粒子：" << endl;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		cout << "站点" << _rowId << "：";
		_graph[_rowId]._stationPassenger->print();
		cout << endl;
	}

	cout << "统计：当前粒子数" << getMovingNum() << " 总粒子数：" << _allSum << endl;
}
int myGraph::getMovingNum()
{
	int temp = 0;
	for (_rowId = 0; _rowId < _stationNum; _rowId++)
	{
		temp = temp + _graph[_rowId]._stationPassenger->_passengerNum;
	}
	return temp;
}
double myGraph::congestedRate()
{
	double tempRate;
	tempRate = getMovingNum() / _allSum;
	return tempRate;
}