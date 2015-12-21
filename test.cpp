#include<iostream>
#include<algorithm>
#include<string>
#include<sstream>
#include<fstream>
#include "BPNN.h"
using namespace std;
void btol(char *p)
{
	swap(p[0], p[3]);
	swap(p[1], p[2]);
}

int main()
{
	vector<vector<double>> data;
	int row, column, data_size, trash;
	uint8_t pixel;
	ifstream in;
	in.open("t10k-images.idx3-ubyte", ios::binary);
//	string line;
//	while(getline(in, line))
//	{
//		vector<double> tmp;
//		int ind;
//		istringstream iss(line);
//		iss>>ind;
//		index.push_back(ind);
//		double d;
//		while(iss>>d)
//			tmp.push_back(d);
//		data.push_back(tmp);
//	}
	in.read((char*)&data_size, 4);
	in.read((char*)&data_size, 4);
	btol((char*)&data_size);
	in.read((char*)&row, 4);
	btol((char*)&row);
	in.read((char*)&column, 4);
	btol((char*)&column);
	for(int i = 0; i < data_size; i++)
	{
		vector<double> tmp;
		for(int j = 0; j < row*column; j++)
		{
			in.read((char*)&pixel, sizeof(pixel));
			tmp.push_back((double)pixel / 255.0);
		}
		data.push_back(tmp);
	}
	in.close();
	in.open("t10k-labels.idx1-ubyte", ios::binary);
	in.read((char*)&trash, 4);
	in.read((char*)&trash, 4);
	vector<int> index;
	for(int i = 0; i < data_size; i++)
	{
		in.read((char*)&pixel, sizeof(pixel));
		index.push_back((int)pixel);
	}

	vector<vector<double>> out;
	for(int i = 0; i < data_size; i++)
	{
		vector<double> t;
		t.resize(10, 0.1);
		t[index[i]] = 0.9;
		out.push_back(t);
	}

	/*
	BPNN bpnn(row * column, 6, 10);
	bpnn.set_learn_rate(0.3);
	bpnn.learn_all(data, out, 5);
	ofstream fout;
	fout.open("learn.net", ios::binary);
	bpnn.save(fout);
	fout.close();
	*/
	ifstream fin;
	fin.open("learn.net", ios::binary);
	BPNN bpnn;
	bpnn.read(fin);
	fin.close();

	int s = 0;
	for(int i = 0; i < (int)data.size(); i++)
	{
		auto vd = bpnn.compute(data[i]);
		auto max_index = [&](vector<double> v)
		{
			double max_ele = -1;
			int max_index = 0;
			for(int j = 0; j < (int) v.size(); ++j)
				if(max_ele < v[j])
					max_ele = v[j], max_index = j;
			return max_index;
		};
		if(max_index(vd)==index[i]) s++;
	}
	cout<<s<<endl;
	return 0;
}
