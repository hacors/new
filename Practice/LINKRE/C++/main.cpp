#pragma once
#include "myGraph.h"
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;
static int mynum = 120;
static float myrate = 0.5;

void text1()
{
	myGraph text(mynum, myrate);
	int temp = mynum * 2;
	while (temp--)
		text.addRoute();
	text.printGraph();
	text.initialDistance();
	text.clearDistance(false);
	text.updatePriorNew();
	text.printDistance();
}
void text2()
{
	myGraph text(1858, 0.5);
	text.establishByInput();
	text.initialDistance();

	text.clearDistance(false);
	text.findNextFloyd();
	text.printDistance();

	text.clearDistance(true);
	text.findNext();
	text.printDistance();
}
void text3()
{
	myGraph text(mynum, myrate);
	text.establishByBA(3);
	text.printGraph();
	text.printDistance();

	text.initialDistance();
	text.printDistance();

	text.updatePriorLP();
	text.updatePriorNew();
	text.printDistance();

	text.clearDistance(false);
	text.findNextFloyd();
	text.printDistance();

	text.clearDistance(true);
	text.findNextDij();
	text.printDistance();

	text.graphBreak(0.5);
	text.printGraph();
	text.printBreakList();

	text.setPriority(type::LP);
	text.printBreakList();
	text.setPriority(type::RSA);
	text.printBreakList();

	text.clearDistance(false);
	text.printDistance();
}
void text4()
{
	myGraph text(mynum, myrate);
	text.establishByBA(3);
	text.initialDistance();
	text.clearDistance(true);
	text.findNextDij();

	text.updateDistance();
	text.updateLoad();
	text.updatePriorLP();
	text.updatePriorNew();

	text.graphBreak(0.5);
	text.setPriority(type::RLP);
	int rounds;
	while (!text._recoveryList.empty())
	{
		text.graphRecovery();
		text.clearDistance(false);
		text.findNextFloyd();
		//text.printDistance();
		rounds = 100;
		text._allSum = 0;
		text.clearLoad();
		while (rounds--)
		{
			text.addLoad(0.5);
			text.allTakeMove();
		}
		//text.printGraph();
		//text.printMovingList();
		text._congestedRate = text.getMovingNum() * 10000 / text._allSum;
		cout << text._congestedRate << " 统计：当前粒子数" << text.getMovingNum() << " 总粒子数：" << text._allSum << endl;
	}
}
int main()
{
	srand((unsigned int)time(0));
	text4();
	return 0;
}