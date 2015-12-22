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
namespace kedixa{

class BPNN
{
	typedef std::vector<double> vd;
	typedef std::vector<vd>     vvd;
private:
	int    num_in;            // 输入层向量维度
	int    num_hid;           // 中间层向量维度
	int    num_out;           // 输出层向量维度
	vd     vec_in;            // 输入层向量
	vd     vec_hid;           // 中间层向量
	vd     vec_out;           // 输出层向量
	vd     delta_out;         // 输出层误差
	vd     delta_hid;         // 中间层误差
	vd     const_in;          // 常数项权值
	vd     const_hid;         //
	vvd    in_hid;            // 输入层到中间层的权值
	vvd    pre_in_hid;        // 上一次的调整量
	vvd    hid_out;           // 中间层到输出层的权值
	vvd    pre_hid_out;
	double learn_rate;        // 学习速率
	double momentum;          // 冲量项
	bool   initialized;       // 参数是否初始化
	bool   init_cap();
	bool   init_weight();
	double sigmoid(double);   // 映射函数
	double sigmoid_d(double); // 映射函数导数

public:
	// 构造函数，参数为各向量维度
	BPNN();
	BPNN(int, int, int);
	bool   set_learn_rate(double);
	bool   set_momentum(double);
	double learn(const vd&, const vd&);
	double learn_all(const vvd&, const vvd&, int times = 100);
	const  vd& compute(const vd&);
	void   save(std::ostream&);
	bool   read(std::istream&);
	bool   clean(); // 释放内存空间
	~BPNN();
};

} // namespace kedixa
#endif // BPNN_H_
