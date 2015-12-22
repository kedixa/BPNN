#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<algorithm>
#include "BPNN.h"
using namespace std;
using namespace kedixa;

vector<vector<double>> input;
vector<vector<double>> target;

void test_number()
{
	// read data
	const int row = 17, column = 11, data_size = 10;
	const int hid_size = 25, out_size= 10;
	input.clear();
	target.clear();
	ifstream in("number.data");
	string s, number[data_size];
	for(int i = 0; i < data_size; ++i)
		for(int j = 0; j < row; ++j)
		{
			in >> s;
			number[i] += s;
		}
	in.close();
	input.resize(data_size);
	for(int i = 0; i < data_size; ++i)
		for(int j = 0; j < row * column; ++j)
			input[i].push_back(number[i][j]=='1'?1:0);
	target.resize(data_size);
	for(int i = 0; i < data_size; ++i)
		target[i].resize(out_size, 0.1), target[i][i] = 0.9;
	BPNN bpnn(row * column, hid_size, out_size);
	bpnn.set_momentum(0.3);
	bpnn.set_learn_rate(0.5);
	// 学习20次
	cout<<"index\terror\n";
	bpnn.learn_all(input, target, 20);
	int correct = 0;
	for(int i = 0; i < data_size; ++i)
	{
		auto x = bpnn.compute(input[i]);
		if(x[i] == *max_element(x.begin(), x.end()))
			correct++;
	}
	cout<<"Accuracy:\t"<<correct<<"/"<<data_size<<endl;
}

void test_iris()
{
	input.clear();
	target.clear();
	vector<int> index;
	const int hid_size = 2, out_size = 3;
	const int data_size = 150, in_size = 4;
	double high = 0.9, low = 0.1;
	target = {{high, low, low}, {low, high, low}, {low, low, high}};

	ifstream in("iris.data");
	double d1, d2, d3, d4;
	int ind;
	for(int i = 0; i < data_size; ++i)
	{
		in >> ind >> d1 >> d2 >> d3 >> d4;
		input.push_back({d1, d2, d3, d4});
		index.push_back(ind);
	}
	in.close();
	BPNN bpnn(in_size, hid_size, out_size);
	bpnn.set_momentum(0.3);
	bpnn.set_learn_rate(0.4);
	// 学习一半数据集50次
	cout<<"index\terror\n";
	for(int i = 0; i < 50; i++)
	{
		double sumerr = 0;
		for(int j = 0; j < data_size / 6; ++j)
		{
			sumerr += bpnn.learn(input[j], target[index[j] - 1]);
			sumerr += bpnn.learn(input[j + 50], target[index[j + 50] - 1]);
			sumerr += bpnn.learn(input[j + 100], target[index[j + 100] - 1]);
		}
		cout<<i<<"\t"<<sumerr<<endl;
	}
	// 测试另一半数据集
	int correct = 0;
	for(int i = 0; i < data_size / 6; ++i)
		for(int j = 0; j < 3; ++j)
		{
			int r = i + 25 + 50 * j;
			auto x = bpnn.compute(input[r]);
			if(x[index[r] - 1] == *max_element(x.begin(), x.end()))
				correct++;
		}
	cout<<"Accuracy:\t"<<correct<<"/"<<data_size/2<<endl;
}

void test_mnist()
{
	input.clear();
	target.clear();
	vector<int> index;
	auto btol = [&](char *p) {
		swap(p[0], p[3]), swap(p[1], p[2]); };
	int row, column, magic, data_size, trash;
	int hid_size = 20, out_size = 10;
	double low = 0.1, high = 0.9;
	uint8_t pixel;
	// 训练集 60000数据
	ifstream in;
	in.open("train-images.idx3-ubyte", ios::binary);
	assert(!in.fail());
	// 读取图像数据
	in.read((char*)&magic, 4); 
	btol((char*)&magic);
	assert(magic == 2051);
	in.read((char*)&data_size, 4);
	btol((char*)&data_size);
	in.read((char*)&row, 4);
	btol((char*)&row);
	in.read((char*)&column, 4);
	btol((char*)&column);
	for(int i = 0; i < data_size; i++)
	{
		vector<double> tmp;
		for(int j = 0; j < row * column; j++)
		{
			in.read((char*)&pixel, sizeof(pixel));
			tmp.push_back((double)pixel / 255.0);
		}
		input.push_back(tmp);
	}
	in.close();
	// 读取label
	in.open("train-labels.idx1-ubyte", ios::binary);
	assert(!in.fail());
	in.read((char*)&magic, 4);
	btol((char*)&magic);
	assert(magic == 2049);
	in.read((char*)&trash, 4);
	btol((char*)&trash);
	assert(trash == data_size);
	for(int i = 0; i < data_size; ++i)
	{
		in.read((char*)&pixel, sizeof(pixel));
		index.push_back((int)pixel);
	}
	in.close();
	// 构造目标输出
	for(int i = 0; i < 10; i++)
	{
		vector<double> t;
		t.resize(10, low);
		t[i] = high;
		target.push_back(t);
	}

	// 学习训练数据
	BPNN bpnn(row * column, hid_size, out_size);
	bpnn.set_learn_rate(0.6);
	bpnn.set_momentum(0.4);
	cout<<"index\terror\n";
	for(int i = 0; i < 30; i++)
	{
		double sumerr = 0;
		for(int j = 0; j < data_size; ++j)
			sumerr += bpnn.learn(input[j], target[index[j]]);
		cout<<i<<"\t"<<sumerr<<endl;
	}
	// 测试数据
	input.clear();
	index.clear();
	in.open("t10k-images.idx3-ubyte", ios::binary);
	assert(!in.fail());
	// 读取图像数据
	in.read((char*)&magic, 4); 
	btol((char*)&magic);
	assert(magic == 2051);
	in.read((char*)&data_size, 4);
	btol((char*)&data_size);
	in.read((char*)&row, 4);
	btol((char*)&row);
	in.read((char*)&column, 4);
	btol((char*)&column);
	for(int i = 0; i < data_size; i++)
	{
		vector<double> tmp;
		for(int j = 0; j < row * column; j++)
		{
			in.read((char*)&pixel, sizeof(pixel));
			tmp.push_back((double)pixel / 255.0);
		}
		input.push_back(tmp);
	}
	in.close();
	// 读取label
	in.open("t10k-labels.idx1-ubyte", ios::binary);
	assert(!in.fail());
	in.read((char*)&magic, 4);
	btol((char*)&magic);
	assert(magic == 2049);
	in.read((char*)&trash, 4);
	btol((char*)&trash);
	assert(trash == data_size);
	for(int i = 0; i < data_size; ++i)
	{
		in.read((char*)&pixel, sizeof(pixel));
		index.push_back((int)pixel);
	}
	in.close();
	int correct = 0;
	for(int i = 0; i < data_size; ++i)
	{
		auto x = bpnn.compute(input[i]);
		if(x[index[i]] == *max_element(x.begin(), x.end()))
			correct++;
	}
	cout<<"Accuracy:\t"<<correct<<"/"<<data_size<<endl;
}

int main()
{
//	test_number();
//	test_iris();
	test_mnist();
	return 0;
}
