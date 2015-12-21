/*
 * Copyright (C) Kedixa Liu
 * 	kedixa@outlook.com
 * 反向传播算法学习
 *
 */

#include "BPNN.h"

/*
 * function: BPNN::BPNN 构造函数
 * _in: 输入层向量维度
 * _hid: 中间层向量维度
 * _out: 输出层向量维度
 *
 */
BPNN::BPNN(int _in, int _hid = 1, int _out = 1)
{
	num_in = _in;
	num_hid = _hid;
	num_out = _out;
	learn_rate = 0.1;

	// 为各个向量预分配空间
	vec_in.resize(num_in);
	vec_hid.resize(num_hid);
	vec_out.resize(num_out);
	delta_hid.resize(num_hid);
	delta_out.resize(num_out);
	in_hid.resize(num_hid);
	hid_out.resize(num_out);
	for(auto& t : in_hid)
		t.resize(num_in);
	for(auto& t : hid_out)
		t.resize(num_hid);
	init();
}

/*
 * function: BPNN::init 初始化权值矩阵参数
 *
 */
bool BPNN::init()
{
	auto rd = [](){return 0.05;};
	for(auto& t : in_hid)
		for(auto& x : t)
			x = rd();
	for(auto& t : hid_out)
		for(auto& x : t)
			x = rd();
	return true;
}

/*
 * function: BPNN::sigmoid
 *
 */
double BPNN::sigmoid(double x)
{
	return 1.0 / (1 + std::exp(-x));
}

double BPNN::sigmoid_d(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

/*
 * function: BPNN::clean 释放内存空间
 *
 */
bool BPNN::clean()
{
	vec_in.clear();
	vec_hid.clear();
	vec_out.clear();
	delta_hid.clear();
	delta_out.clear();
	in_hid.clear();
	hid_out.clear();
	return true;
}

BPNN::~BPNN()
{
	clean();
}

/*
 * function: BPNN::set_learn_rate 设置学习速率
 *
 */
bool BPNN::set_learn_rate(double _rate)
{
	if(_rate < 0.001 || _rate > 1.0)
		return false;
	learn_rate = _rate;
	return true;
}

/*
 * function: BPNN::compute 根据输入向量计算输出向量
 * _in: 输入向量
 * return: 输出层向量的引用
 * 
 */
const BPNN::vd& BPNN::compute(const vd& _in)
{
	assert((int)_in.size() >= num_in);
	std::copy_n(_in.begin(), num_in, vec_in.begin());
	std::fill(vec_hid.begin(), vec_hid.end(), 0.0);
	std::fill(vec_out.begin(), vec_out.end(), 0.0);
	// 权值矩阵乘以输入向量得到输出向量
	for(int i = 0; i < num_hid; ++i)
		for(int j = 0; j < num_in; ++j)
			vec_hid[i] += in_hid[i][j] * vec_in[j];
	for(int i = 0; i < num_hid; ++i)
		vec_hid[i] = sigmoid(vec_hid[i]);

	for(int i = 0; i < num_out; ++i)
		for(int j = 0; j < num_hid; ++j)
			vec_out[i] += hid_out[i][j] * vec_hid[j];
	for(int i = 0; i < num_out; ++i)
		vec_out[i] = sigmoid(vec_out[i]);
	return vec_out;
}

/*
 * function: BPNN::learn 根据输入和目标输出进行学习
 * _in: 输入向量
 * out: 目标输出向量
 *
 */

bool BPNN::learn(const vd& _in, const vd& out)
{
	// 首先计算
	compute(_in);
	// 根据计算结果更新权值
	// 计算误差项
	for(int i = 0; i < num_out; ++i)
		delta_out[i] = vec_out[i] * (1 - vec_out[i]) * (out[i] - vec_out[i]);
//	for(auto& i : delta_out)
//		std::cout<<i<<' ';
//	std::cout<<std::endl;
	for(int i = 0; i < num_hid; ++i)
	{
		delta_hid[i] = 0;
		for(int j = 0; j < num_out; ++j)
			delta_hid[i] += hid_out[j][i] * delta_out[j];
		delta_hid[i] *= vec_hid[i] * (1 - vec_hid[i]);
	}
//	for(auto& i :delta_hid)
//		std::cout<<i<<' ';
//	std::cout<<'\n';
	// 更新网络权值
	for(int i = 0; i < num_out; ++i)
		for(int j = 0; j < num_hid; ++j)
			hid_out[i][j] += learn_rate * delta_out[i] * vec_hid[j];
	for(int i = 0; i < num_hid; ++i)
		for(int j = 0; j < num_in; ++j)
			in_hid[i][j] += learn_rate * delta_hid[i] * vec_in[j];
	return true;
}

/*
 * function: BPNN::save // 保存学习结果
 *
 */
void BPNN::save(std::ostream& out)
{
	for(int i = 0; i < num_in; ++i)
	{
		for(int j = 0; j < num_hid; ++j)
			out<<in_hid[j][i]<<' ';
		out<<'\n';
	}
}
