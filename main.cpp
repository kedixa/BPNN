#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<algorithm>
#include "BPNN.h"
using namespace std;

const int row = 17;
const int column = 11;
string num[10];
vector<vector<double>> vec;
vector<vector<double>> out;

void read_data()
{
	ifstream in("number.txt");
	string s;
	for(int i = 0; i < 10; i++)
	{
		num[i]="";
		for(int j = 0; j < row; j++)
		{
			in >> s;
			num[i]+=s;
		}
	}
	in.close();
	vec.resize(10);
	for(int i = 0; i < 10; ++i)
	{
		for(int j = 0; j < row * column; j++)
		{
			if(num[i][j]=='1')
				vec[i].push_back(1);
			else vec[i].push_back(0);
		}
	}
	out.resize(10);
	for(int i = 0; i < 10; i++)
	{
		out[i].resize(10);
		fill(out[i].begin(),out[i].end(),0.1);
		out[i][i]=0.9;
	}
}


int main()
{
	read_data();
	BPNN bpnn(row * column, 30, 10);
//	int from = 0, to= 9;
//	for(int i = 0; i < 5000; i++)
//	{
//		for(int j = from; j < to; j++)
//			bpnn.learn(vec[j], out[j]);
//	}
	bpnn.learn_all(vec, out, 100);
	ofstream saveout("a.txt");
	bpnn.save(saveout);
	saveout.close();
	return 0;
}
