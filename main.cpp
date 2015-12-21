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
		out[i][i]= 0.9;
	}
}

double distance(vector<double> x, vector<double> y)
{
	double ans = 0;
	for(int i = 0; i <(int)x.size();i++)
		ans += (x[i]-y[i])*(x[i]-y[i]);
	return ans;
}

int main()
{
	read_data();
		BPNN bpnn(row * column, 6, 10);
		bpnn.set_learn_rate(0.3);
		for(int i = 0; i < 200; i++)
		{
			for(int j = 0; j < 10; j++)
				bpnn.learn(vec[j], out[j]);
		}
//		bpnn.learn_all(vec, out, 1000);

		for(int i = 0; i < 10; i++)
		{
			auto x = bpnn.compute(vec[i]);
			int best_index=0;
			double min_val = 999;
			for(int j = 0; j < 10;j ++)
			{
				cout<<x[j]<<' ';
				double dist = distance(x, out[j]);
				if(min_val > dist)min_val = dist,best_index = j;
			}

			cout<<best_index<<"\n";
		}
		cout<<endl;
	return 0;
}
