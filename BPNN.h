/*
 * Copyright (C) Kedixa Liu
 * 	kedixa@outlook.com
 * 反向传播算法学习
 *
 */

#include<iostream>
#include<vector>
#include<algorithm>
#include<cassert>
#include<cmath>

#ifndef BPNN_H_
#define BPNN_H_

class BPNN
{
	typedef std::vector<double> vd;
	typedef std::vector<vd> vvd;
private:
	int num_in; // 输入层向量维度
	int num_hid; // 中间层向量维度
	int num_out; // 输出层向量维度
	vd vec_in; // 输入层向量
	vd vec_hid; // 中间层向量
	vd vec_out; // 输出层向量
	vd delta_out; // 输出层误差
	vd delta_hid; // 中间层误差
	vd const_in; // 常数项权值
	vd const_hid; //
	vvd in_hid; // 输入层到中间层的权值
	vvd hid_out; // 中间层到输出层的权值
	double learn_rate;
	bool init();
	bool clean();
	double sigmoid(double); // 映射函数
	double sigmoid_d(double); // 映射函数导数

public:
	// 构造函数，参数为各向量维度
	BPNN(int, int, int);
	bool set_learn_rate(double);
	bool learn(const vd&, const vd&);
	bool learn_all(const vvd&, const vvd&, int times = 100);
	const vd& compute(const vd&);
	void save(std::ostream&);
	~BPNN();
};

#endif // BPNN_H_
