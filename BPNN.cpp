/*
 * Copyright (C) Kedixa Liu
 * 	kedixa@outlook.com
 * 反向传播算法学习
 *
 */

#include "BPNN.h"
#include<random>
namespace kedixa{

/*
 * function: BPNN::BPNN 构造函数
 * _in: 输入层向量维度
 * _hid: 中间层向量维度
 * _out: 输出层向量维度
 *
 */
BPNN::BPNN()
{
	initialized = false;
}

BPNN::BPNN(int _in, int _hid = 1, int _out = 1)
{
	num_in     = _in;
	num_hid    = _hid;
	num_out    = _out;
	learn_rate = 0.1;
	momentum   = 0.3;

	init_cap();
	init_weight();
	initialized = true;
}

/*
 * function:: BPNN::init_cap 
 *
 */
bool BPNN::init_cap()
{
	// 为各个向量预分配空间
	vec_in.resize(num_in);
	vec_hid.resize(num_hid);
	vec_out.resize(num_out);
	delta_hid.resize(num_hid);
	delta_out.resize(num_out);
	const_in.resize(num_hid);
	const_hid.resize(num_out);
	in_hid.resize(num_hid);
	hid_out.resize(num_out);
	pre_in_hid.resize(num_hid);
	pre_hid_out.resize(num_out);
	for(auto& t : in_hid)
		t.resize(num_in);
	for(auto&t : pre_in_hid)
		t.resize(num_in);
	for(auto& t : hid_out)
		t.resize(num_hid);
	for(auto& t : pre_hid_out)
		t.resize(num_hid);
	return true;
}

/*
 * function: BPNN::init_weight 初始化权值矩阵参数
 *
 */
bool BPNN::init_weight()
{
	std::random_device r;
	// 产生 -0.25 ～0.25 的随机数
	// 开始因为懒，设置为恒定值，导致效果差，调试了一天，特此留念
	auto rd = [&](){return (double)r()/ 2.0 / r.max() - 0.25;};
	for(auto& t : const_in)
		t = rd();
	for(auto& t : const_hid)
		t = rd();
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
inline double BPNN::sigmoid(double x)
{
	// 如果要更改此函数，同时应该更改函数导数
	return 1.0 / (1 + std::exp(-x));
}

/*
 * function: BPNN::sigmoid_d // sigmoid的导数
 *
 */
inline double BPNN::sigmoid_d(double x)
{
	return x * (1 - x);
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
	pre_in_hid.clear();
	pre_hid_out.clear();
	const_in.clear();
	const_hid.clear();
	initialized = false;
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
 * function: BPNN::set_momentum 设置冲量项
 *
 */
bool BPNN::set_momentum(double _mom)
{
	if(_mom < 0 || _mom > 1.0)
		return false;
	momentum = _mom;
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
	if(!initialized) exit(1); // 未初始化
	assert((int)_in.size() >= num_in);
	std::copy_n(_in.begin(), num_in, vec_in.begin());
	std::fill(vec_hid.begin(), vec_hid.end(), 0.0);
	std::fill(vec_out.begin(), vec_out.end(), 0.0);
	// 权值矩阵乘以输入向量得到输出向量
	for(int i = 0; i < num_hid; ++i)
		for(int j = 0; j < num_in; ++j)
			vec_hid[i] += in_hid[i][j] * vec_in[j];
	for(int i = 0; i < num_hid; ++i)
		vec_hid[i] = sigmoid(vec_hid[i] + const_in[i]);

	for(int i = 0; i < num_out; ++i)
		for(int j = 0; j < num_hid; ++j)
			vec_out[i] += hid_out[i][j] * vec_hid[j];
	for(int i = 0; i < num_out; ++i)
		vec_out[i] = sigmoid(vec_out[i] + const_hid[i]);
	return vec_out;
}

/*
 * function: BPNN::learn 根据输入和目标输出进行学习
 * _in: 输入向量
 * out: 目标输出向量
 *
 */

double BPNN::learn(const vd& _in, const vd& out)
{
	// 首先计算
	compute(_in);
	// 根据计算结果更新权值
	// 计算误差项
	double error=0;
	for(int i = 0; i < num_out; ++i)
		delta_out[i] = sigmoid_d(vec_out[i]) * (out[i] - vec_out[i]),
			error += std::abs(delta_out[i]);
	for(int i = 0; i < num_hid; ++i)
	{
		delta_hid[i] = 0;
		for(int j = 0; j < num_out; ++j)
			delta_hid[i] += hid_out[j][i] * delta_out[j];
		delta_hid[i] *= sigmoid_d(vec_hid[i]);
		error+=std::abs(delta_hid[i]);
	}
	// 更新网络权值
	double d_ij;
	for(int i = 0; i < num_out; ++i)
		for(int j = 0; j < num_hid; ++j)
		{
			d_ij = learn_rate * delta_out[i] * vec_hid[j] + momentum * pre_hid_out[i][j];
			hid_out[i][j] += d_ij;
			pre_hid_out[i][j] = d_ij;
		}
	for(int i = 0; i < num_out; ++i)
		const_hid[i] += learn_rate * delta_out[i];

	for(int i = 0; i < num_hid; ++i)
		for(int j = 0; j < num_in; ++j)
		{
			d_ij = learn_rate * delta_hid[i] * vec_in[j] + momentum * pre_in_hid[i][j];
			in_hid[i][j] += d_ij;
			pre_in_hid[i][j] = d_ij;
		}
	for(int i = 0; i < num_hid; ++i)
		const_in[i] += learn_rate * delta_hid[i];

	return error;
}

/*
 * function: BPNN::learn_all 学习数据集
 * return: 最后一遍学习的误差值
 *
 */
double BPNN::learn_all(const vvd& _in, const vvd& _out, int times)
{
	double sumerr = 0;
	for(int i = 0; i < times; ++i)
	{
		sumerr = 0;
		for(int j = 0; j < (int)_in.size(); ++j)
			sumerr += learn(_in[j], _out[j]);
		std::cout << i << ":\t" << sumerr << std::endl;
	}
	return sumerr;
}

/*
 * function: BPNN::save // 保存学习结果
 *
 */
void BPNN::save(std::ostream& out)
{
	int magic_number = 0x12345678;
	out.write((char*)&magic_number, sizeof(int));
	out.write((char*)&num_in, sizeof(int));
	out.write((char*)&num_hid, sizeof(int));
	out.write((char*)&num_out, sizeof(int));
	out.write((char*)&learn_rate, sizeof(double));
	out.write((char*)&momentum, sizeof(double));
	auto write_vd = [&](vd& v)
	{
		for(auto& x : v)
			out.write((char*)&x, sizeof(double));
	};
	auto write_vvd = [&](vvd& v)
	{
		for(auto& x : v)
			write_vd(x);
	};
	write_vd(vec_in);
	write_vd(vec_hid);
	write_vd(vec_out);
	write_vd(const_in);
	write_vd(const_hid);
	write_vd(delta_hid);
	write_vd(delta_out);
	write_vvd(in_hid);
	write_vvd(pre_in_hid);
	write_vvd(hid_out);
	write_vvd(pre_hid_out);
	out.flush();
}

/*
 * function: BPNN::read // 读取学习结果
 *
 */
bool BPNN::read(std::istream& in)
{
	int magic_number;
	in.read((char*)&magic_number, sizeof(int));
	if(magic_number != 0x12345678) return false;
	in.read((char*)&num_in, sizeof(int));
	in.read((char*)&num_hid, sizeof(int));
	in.read((char*)&num_out, sizeof(int));
	in.read((char*)&learn_rate, sizeof(double));
	in.read((char*)&momentum, sizeof(double));
	assert(num_in > 0);
	assert(num_hid > 0);
	assert(num_out > 0);
	if(!init_cap()) return false;
	auto read_vd = [&](vd& v)
	{
		for(auto&x : v)
			in.read((char*)&x, sizeof(double));
	};
	auto read_vvd = [&](vvd& v)
	{
		for(auto&x : v)
			read_vd(x);
	};
	read_vd(vec_in);
	read_vd(vec_hid);
	read_vd(vec_out);
	read_vd(const_in);
	read_vd(const_hid);
	read_vd(delta_hid);
	read_vd(delta_out);
	read_vvd(in_hid);
	read_vvd(pre_in_hid);
	read_vvd(hid_out);
	read_vvd(pre_hid_out);
	initialized = true;
	return true;
}

} // namespace kedixa
