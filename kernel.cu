#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "kernal.h"
#include "cuda_texture_types.h"//否则不识别texture
//__constant__  int sinTable512[] = {
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250,
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2
//};
//
//__constant__ int cosTable512[] = {
//	250, 250, 250, 250, 250, 249, 249, 249, 249, 248, 248, 248, 247, 247, 246, 245,
//	245, 244, 244, 243, 242, 241, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232,
//	230, 229, 228, 227, 225, 224, 223, 221, 220, 218, 217, 215, 214, 212, 210, 209,
//	207, 205, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178,
//	176, 173, 171, 169, 167, 164, 162, 160, 157, 155, 153, 150, 148, 145, 143, 140,
//	138, 135, 132, 130, 127, 125, 122, 119, 116, 114, 111, 108, 105, 103, 100,  97,
//	94,  91,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
//	47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2,
//	-2,  -5,  -8, -11, -14, -17, -20, -23, -26, -29, -32, -35, -38, -41, -44, -47,
//	-50, -53, -56, -59, -62, -65, -68, -71, -74, -77, -80, -83, -86, -89, -91, -94,
//	-97,-100,-103,-105,-108,-111,-114,-116,-119,-122,-125,-127,-130,-132,-135,-138,
//	-140,-143,-145,-148,-150,-153,-155,-157,-160,-162,-164,-167,-169,-171,-173,-176,
//	-178,-180,-182,-184,-186,-188,-190,-192,-194,-196,-198,-200,-202,-204,-205,-207,
//	-209,-210,-212,-214,-215,-217,-218,-220,-221,-223,-224,-225,-227,-228,-229,-230,
//	-232,-233,-234,-235,-236,-237,-238,-239,-240,-241,-241,-242,-243,-244,-244,-245,
//	-245,-246,-247,-247,-248,-248,-248,-249,-249,-249,-249,-250,-250,-250,-250,-250,
//	-250,-250,-250,-250,-250,-249,-249,-249,-249,-248,-248,-248,-247,-247,-246,-245,
//	-245,-244,-244,-243,-242,-241,-241,-240,-239,-238,-237,-236,-235,-234,-233,-232,
//	-230,-229,-228,-227,-225,-224,-223,-221,-220,-218,-217,-215,-214,-212,-210,-209,
//	-207,-205,-204,-202,-200,-198,-196,-194,-192,-190,-188,-186,-184,-182,-180,-178,
//	-176,-173,-171,-169,-167,-164,-162,-160,-157,-155,-153,-150,-148,-145,-143,-140,
//	-138,-135,-132,-130,-127,-125,-122,-119,-116,-114,-111,-108,-105,-103,-100, -97,
//	-94, -91, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -53, -50,
//	-47, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -11,  -8,  -5,  -2,
//	2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
//	50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  91,  94,
//	97, 100, 103, 105, 108, 111, 114, 116, 119, 122, 125, 127, 130, 132, 135, 138,
//	140, 143, 145, 148, 150, 153, 155, 157, 160, 162, 164, 167, 169, 171, 173, 176,
//	178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 205, 207,
//	209, 210, 212, 214, 215, 217, 218, 220, 221, 223, 224, 225, 227, 228, 229, 230,
//	232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 241, 242, 243, 244, 244, 245,
//	245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 249, 250, 250, 250, 250, 250
//};



//#define SIZE 1024
#define BLOCK_SIZES 1024
#define GRID_SIZES 256
#define L 64

texture<int> t_sinTable;
texture<int> t_cosTable;
texture<int, 1> t_CAcode;
texture<char, 1> t_navbit;
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s error code %d\n", cudaGetErrorString(result), result);
		getchar();
		//assert(result == cudaSuccess);
	}
#endif
	return result;
}



__global__ void kernelMultiArray2D(short* ACos, short* ASin, short* B, short* CCos, short* CSin, int rows, int cols, short gain) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;
	int temp = ACos[index];
	ACos[index] = temp * B[index] * CCos[index] * gain;
	ASin[index] = temp * B[index] * CSin[index] * gain;
}

__global__ void kernerSumColumnArray2D(short* ACos, short* ASin, int rows, int cols, short* iq_buff) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;//(13,1*10^6) 256*1024
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * cols;
	if (col < 2)
	{
		printf("col=%d", col);
		printf("row=%d", row);
	}
	if (col < cols && row == 0) {
		int sumCos = 0;
		int sumSin = 0;
		for (int i = 0; i < rows; i++) {
			sumCos += ACos[i * cols + index];
			sumSin += ASin[i * cols + index];
		}
		iq_buff[col * 2] = short((sumCos + 64) >> 7);
		iq_buff[col * 2 + 1] = short((sumSin + 64) >> 7);

	}
}

//multiArray2D_Wrapper(cos_page, sin_page, dev_CCos, dev_CSin, d_iq_buff, count, iq_buff_size, iq_buff);
void multiArray2D_Wrapper(short* h_A, short* h_B, short* dev_CCos, short* dev_CSin, short* d_iq_buff, int rows, int cols, short* h_iq_buff)
{

	size_t size_array1D = cols * sizeof(short);
	size_t size_array2D = rows * cols * sizeof(short);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int blockSize = BLOCK_SIZES;
	int gridSize = (rows * cols + blockSize - 1) / blockSize;


	checkCuda(cudaMemcpy(dev_CCos, h_A, size_array2D, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_CSin, h_B, size_array2D, cudaMemcpyHostToDevice));//已经乘好的信号





	//kernelMultiArray2D << <gridSize, blockSize >> >(dev_ACos, dev_ASin, dev_B, dev_CCos, dev_CSin, rows, cols, gain);
	kernerSumColumnArray2D << <gridSize, blockSize >> > (dev_CCos, dev_CSin, rows, cols, d_iq_buff);
	//cudathreadsynchronize();

	checkCuda(cudaMemcpy(h_iq_buff, d_iq_buff, cols * 2 * sizeof(short), cudaMemcpyDeviceToHost));
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);

	printf("\nhet\n");
}
//折叠部分为存取纹理内存的核函数

//__global__ void get_textureMem2D(int satnum, int* output)
//{
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;//ch_num=y,sample_num=x
//	int idy = threadIdx.y;
//	/*if (idy == satnum)
//		printf("%d ", tex1D(t_CAcode, (float)idx + (float)idy * 1023));*/
//
//	while (idx < 1023 && idy == satnum)
//	{
//		//float u = (float)idx / 1023;
//		//float v = (float)idy / 10;
//		output[idx] = tex1Dfetch(t_navbit, (idx + idy * 1023));
//		//output[idx] = cacode[idx+idy*1023];
//		idx += blockDim.x * gridDim.x;
//	}
//	//while (idx < 512 && idy == 1)
//	//{
//	//	/*int temp = tex1D(t_cosTable, float(idx));*/
//	//	output[idx] = tex1D(t_cosTable, (float)idx);
//	//	idx += blockDim.x * gridDim.x;
//	//}
//}
/*
* 功能:存入经常使用的查询表和变量
* sin/cos table: texture memory
* pseudorandom code:texture memory
* navigation data:
* i_buff/q_buff:page-locked memory
* amplititude: texture memory
* input: sinTable,cosTable,CAcode
*/
void GPUMemoryInit(Table* Tb, int ch_num)
{
	int* output, * dev_output;
	int satnum = 0;//在和函数中使用，用于检查第satnum颗星星历或prn是否正常
	output = (int*)malloc(sizeof(int) * 1023);

	cudaChannelFormatDesc coschannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_cosTable), &coschannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_cosTable, 0, 0, Tb->cosTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_cosTable, Tb->cu_cosTable);//将余弦表绑定为纹理内存

	//for (int i = 0; i < 1023; i++)
	//{
	//	printf("%d ", *(Tb->CAcode + i + satnum * 1023));
	//}
	//printf("\n");

	cudaChannelFormatDesc sinchannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaMallocArray(&(Tb->cu_sinTable), &sinchannelDesc, 512, 1);
	cudaMemcpyToArray(Tb->cu_sinTable, 0, 0, Tb->sinTable, sizeof(int) * 512, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(t_sinTable, Tb->cu_sinTable);//将正弦表绑定为纹理内存

	output = (int*)malloc(sizeof(int) * 1023);
	checkCuda(cudaMalloc((void**)&(dev_output), sizeof(int) * 1023));

	checkCuda(cudaMalloc((void**)&Tb->dev_CAcode, sizeof(int) * 1023 * MAX_SAT));
	cudaMemcpy(Tb->dev_CAcode, Tb->CAcode, sizeof(int) * 1023 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_CAcode, Tb->dev_CAcode);//将CA码绑定为纹理内存

	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_SAT));//1800比特,一个帧包含5个子帧，一个子帧十个字，一个字30比特，一个比特占一个char,存储的时候把上一帧的第五子帧也保留下来，这样调用的时候方便一点
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//将导航电文绑定为纹理内存

	dim3 block(28, 1);
	dim3 thread(L, ch_num);
	// test initial table is right
	//get_textureMem2D << <block, thread >> > (satnum, dev_output);
	checkCuda(cudaMemcpy(output, dev_output, sizeof(int) * 1023, cudaMemcpyDeviceToHost));

	//for (int j = 0; j < 1023; j++)
	//{
	//	printf("%d ", output[j]);
	//	//printf("%d ", output[j] - *(Tb->CAcode + j + satnum * 1023));
	//	//if ((output[j] - *(Tb->CAcode + j + satnum * 1023)) != 0)
	//}
}

void navdata_update(Table* Tb)
{
	checkCuda(cudaMalloc((void**)&Tb->dev_navdata, sizeof(char) * 1800 * MAX_SAT));
	cudaMemcpy(Tb->dev_navdata, Tb->navdata, sizeof(char) * 1800 * MAX_SAT, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, t_navbit, Tb->dev_navdata);//将导航电文绑定为纹理内存
}

__global__ void cudaBPSK(float* dev_parameters, float* dev_i_buff, int* dev_sum, float* dev_noise,double *dev_db_para)//carrier_step
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x+1;//ch_num=y,sample_num=x idx等于零的时候没有对码相位进行累加，实际上应该累加的
	int idy = threadIdx.y;
	int prn = dev_parameters[idy*p_n+2]-1;//在表中prn是从0开始的
	double codephase = (idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]);
	codephase -= (int)(codephase);
	int CurrentCodePhase = (int)(idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0])%1023 ;//码相位=采样点数*码相位步进+初始码相位 可能会超出1023，所以需要与1023取余
	codephase+= CurrentCodePhase;
	double CarrierPhase = (idx * dev_db_para[idy * pd_n + 3] + dev_db_para[idy * pd_n + 2]); //相位的单位是周(2*pi) 载波相位=采样点数*载波相位步进+初始载波相位
	int cph = 0;
	if(CarrierPhase<0)
		 cph = (CarrierPhase - (int)CarrierPhase) * 512+512;//留下小数部分
	else
		 cph = (CarrierPhase - (int)CarrierPhase) * 512;    //留下小数部分  零中频情况下出现了负值
	//int cph = (CarrierPhase >> 16) & 511;
	int temp = (int)(idx * dev_db_para[idy * pd_n + 1] + dev_db_para[idy * pd_n + 0]) / 1023+ dev_parameters[idy * p_n + 4] ;//得到ms数
	int ibit = temp / 20+dev_parameters[idy*p_n+0]+ dev_parameters[idy * p_n + 3]*30;//前300bit为上一个帧的最后一个子帧，从301开始是当前子帧的第一个bit
	__shared__ float memoryi[MAX_CHAN][threadPerBlock];//本机GPU每线程块内的共享内存为48KB
	//if (idy ==0 )
	//{
	//	dev_i_buff[idx - 1] = CarrierPhase;
	//}
	
	if (idx < dev_sum[0])
	{
		memoryi[idy][threadIdx.x] = dev_parameters[p_n * idy + 1] * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 1023))\
			* tex1D(t_cosTable, (float)cph)* tex1Dfetch(t_navbit, (ibit + prn * 1800))/250;
	}
	__syncthreads();
	/*测试CA码/正余弦表是否生成正常*/
	//if(idx<512)
	//dev_i_buff[idx] = tex1Dfetch(t_CAcode, (idx + 2 * 1023));
	//dev_i_buff[idx] = tex1D(t_cosTable, idx);
	//dev_i_buff[idx] = dev_parameters[6 * idy + 5];

	//这里没有使用归约运算
	//if (idy == 0)
	//{
	//	for (char i = 1; i < dev_sum[1]; i++)
	//	{
	//		memoryi[0][threadIdx.x] += memoryi[i][threadIdx.x];
	//	}
	//}
	//dev_i_buff[idx] = memoryi[0][threadIdx.x];
	//这里使用归约运算
	int i = dev_sum[1] / 2;
	while (i !=0)//只进行一次归约
	{
		if (idy < i)
			memoryi[idy][threadIdx.x] += memoryi[idy + i][threadIdx.x];
		__syncthreads();
		i /= 2;
	}
	//if(idy==1)
	//int num1 = idx * 2;
	if (idy == 0)
	{
		int j = dev_noise[idx];
		dev_i_buff[idx] = memoryi[0][threadIdx.x]+ dev_noise[idx];  //short((memoryi[0][threadIdx.x] + 64) >> 7)   dev_noise[idx]
		//dev_i_buff[idx] = dev_noise[idx];
	}

	//计算另一路的数值
	if (idx < dev_sum[0])
	{
		memoryi[idy][threadIdx.x] = dev_parameters[p_n * idy + 1] * tex1Dfetch(t_CAcode, (CurrentCodePhase + prn * 1023))\
			* tex1D(t_sinTable, (float)cph) * tex1Dfetch(t_navbit, (ibit + prn * 1800))/250;
	}
	__syncthreads();
	i = dev_sum[1] / 2;
	while (i !=0)
	{
		if (idy < i)
			memoryi[idy][threadIdx.x] += memoryi[idy + i][threadIdx.x];
		__syncthreads();
		i /= 2;
	}
	//num1 -= 1;
	if (idy == 0)
	{
		int j = dev_noise[idx + dev_sum[0]];
		dev_i_buff[idx + dev_sum[0]] = memoryi[0][threadIdx.x]+dev_noise[idx+dev_sum[0]]; //+dev_noise[idx+dev_sum[0]]
		//dev_i_buff[idx + dev_sum[0]] = dev_noise[idx + dev_sum[0]];
	}
}



void produce_samples_withCuda(Table* Tb, channel_t* channel, int fs, float* parameters, float* dev_parameters, int* sum, int* dev_sum, float* dev_i_buff, int satnum, float* dev_noise, double* db_para, double* dev_db_para)
{
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	int sat_all = satnum;
	float * dev_amplitude, * amplitude;
	int samples = fs / Rev_fre;
	float* dev_buff;
	for (int j = 0; j < sat_all; j++, channel++)
	{
		if (channel->prn != 0)
		{
			parameters[j * p_n + 0] = channel->ibit;
			parameters[j * p_n + 1] = channel->amp;
			parameters[j * p_n + 2] = channel->prn;
			parameters[j * p_n + 3] = channel->iword;
			parameters[j * p_n + 4] = channel->icode;
			db_para[j * pd_n + 0] = channel->code_phase;
			db_para[j * pd_n + 1] = channel->code_phasestep;
			db_para[j * pd_n + 2] = channel->carr_phase;
			db_para[j * pd_n + 3] = channel->carr_phasestep;
		}
	}

	
	//
	checkCuda(cudaMalloc((void**)&dev_buff, samples * sizeof(float) * 2));
	checkCuda(cudaMemcpy(dev_parameters, parameters, sizeof(float) * sat_all * p_n, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_db_para, db_para, sizeof(double) * sat_all * pd_n, cudaMemcpyHostToDevice));

	sum[0] = samples; sum[1] = MAX_CHAN;
  	checkCuda(cudaMemcpy(dev_sum, &sum[0], 2 * sizeof(int), cudaMemcpyHostToDevice));
	float blockPerGridx = fs / Rev_fre / threadPerBlock;
	(blockPerGridx - (int)blockPerGridx == 0) ? blockPerGridx : blockPerGridx = (int)blockPerGridx + 1;
	dim3 block(blockPerGridx, 1);
	dim3 thread(threadPerBlock, satnum);
	cudaBPSK << <block, thread >> > (dev_parameters, dev_buff, dev_sum,dev_noise,dev_db_para);

	checkCuda(cudaMemcpy(Tb->i_buff, dev_buff, sizeof(float) * samples * 2, cudaMemcpyDeviceToHost));

	cudaFree(dev_buff);
	//printf("\n");
	//for (int i = 0; i < 1000; i++)
	//{
	//	printf("%f\n", Tb->i_buff[i]);
	//}
	//printf("\n");


	//for (int i = 0; i < 10; i++)
	//{
	//	printf("%.5f ", Tb->q_buff[i]);
	//}
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//float elasptime;
	//cudaEventElapsedTime(&elasptime, start, stop);//计算cuda运行时间
	//printf("      \ntime=%.5f ms\n", elasptime);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
}

__global__ void kernalQuantify(float* iq_buff, unsigned char* out_buff, int iq_buff_size, float th)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float src = iq_buff[idx];
	int bcount = idx / 4;
	unsigned char ibit = ibit & 0x00;
	__shared__ unsigned char memory[16];
	if (src > th)
	{
		ibit = 0x01;//3
	}
	else if (src < th && src>0)
	{
		ibit = 0x00;//1
	}
	else if (src < -th)
	{
		ibit = 0x03;//-3
	}
	else if (src > -th && src < 0)
	{
		ibit = 0x02;//-1
	}
	memory[threadIdx.x] = ibit;
	__syncthreads();
	if (threadIdx.x % 4 == 0)
		out_buff[bcount] = memory[threadIdx.x] << 6 | memory[1 + threadIdx.x] << 4 | memory[2 + threadIdx.x] << 2 | memory[3 + threadIdx.x];
}
void quantify_with_cuda(float* dev_iq_buff, float* iq_buff, unsigned char* out_buff, unsigned char* dev_out_buff, int iq_buff_size, float* th)
{
	checkCuda(cudaMemcpy(dev_iq_buff, iq_buff, sizeof(float) * iq_buff_size * 2, cudaMemcpyHostToDevice));
	int blockxPergrid = iq_buff_size * 2 / 16 + 1;
	dim3 block(blockxPergrid, 1);
	dim3 thread(16, 1);
	kernalQuantify << <block, thread >> > (dev_iq_buff, dev_out_buff, iq_buff_size, *th);

	checkCuda(cudaMemcpy(out_buff, dev_out_buff, iq_buff_size / 2, cudaMemcpyDeviceToHost));
}

void GPUMemroy_delete(Table* Tb)
{
	cudaUnbindTexture(t_sinTable);
	cudaUnbindTexture(t_cosTable);
	cudaUnbindTexture(t_CAcode);
	cudaFree(Tb->dev_CAcode);
	cudaFreeArray(Tb->cu_cosTable);
	cudaFreeArray(Tb->cu_sinTable);
	//free(Tb->i_buff);
}





//__global__ void kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if (idx < iq_buff_size) {
//		int ip, qp, i_acc, q_acc;
//		int iTable;
//		i_acc = 0;
//		q_acc = 0;
//		for (int i = 0; i < count; i++) {
//			if (chan[i].prn > 0) {
//
//				iTable = (chan[i].carr_phase >> 16) & 511;
//
//				ip = chan[i].dataBit * chan[i].codeCA * cosTable512[iTable] * gain[i];
//				qp = chan[i].dataBit * chan[i].codeCA * sinTable512[iTable] * gain[i];
//
//				i_acc += ip;
//				q_acc += qp;
//
//				chan[i].code_phase += chan[i].f_code * delt;
//
//				if (chan[i].code_phase >= CA_SEQ_LEN) {
//
//					chan[i].code_phase -= CA_SEQ_LEN;
//					chan[i].icode++;
//
//					if (chan[i].icode >= 20) { // 20 C/A codes = 1 navigation data bit
//						chan[i].icode = 0;
//						chan[i].ibit++;
//
//						if (chan[i].ibit >= 30) { // 30 navigation data bits = 1 word
//							chan[i].ibit = 0;
//							chan[i].iword++;
//							/*
//							if (chan[i].iword>=N_DWRD)
//							printf("\nWARNING: Subframe word buffer overflow.\n");
//							*/
//						}
//
//						// Set new navigation data bit
//						chan[i].dataBit = (int)((chan[i].dwrd[chan[i].iword] >> (29 - chan[i].ibit)) & 0x1UL) * 2 - 1;
//					}
//				}
//
//				// Set currnt code chip
//				chan[i].codeCA = chan[i].ca[(int)chan[i].code_phase] <<1- 1;
//
//				// Update carrier phase
//				chan[i].carr_phase += chan[i].carr_phasestep;
//			}
//		}
//
//
//		// Scaled by 2^7
//		i_acc = (i_acc + 64) >> 7;
//		q_acc = (q_acc + 64) >> 7;
//
//		// Store I/Q samples into buffer
//		iq_buff[idx * 2] = (short)i_acc;
//		iq_buff[idx * 2 + 1] = (short)q_acc;
//	}
//}
//
//extern "C" void handleData_in_kernel(channel_t *chan, int *gain, double delt, int count, int iq_buff_size, short *iq_buff) {
//	
//	size_t sizeChannel = count * sizeof(channel_t);
//	size_t sizeIq_buff = iq_buff_size * 2 * sizeof(short);
//	size_t sizeGain = count * sizeof(int);
//	
//	int *dev_gain;
//	short *dev_iq_buff;
//	channel_t *dev_chan;
//
//	int blockSize = 1024;
//	int gridSize = (iq_buff_size + blockSize - 1) / blockSize;
//
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start);
//
//	checkCuda(cudaMalloc((void**)&dev_chan, sizeChannel));
//	checkCuda(cudaMalloc((void**)&dev_gain, sizeGain));
//	checkCuda(cudaMalloc((void**)&dev_iq_buff, sizeIq_buff));
//
//	checkCuda(cudaMemcpy(dev_gain, gain, sizeGain, cudaMemcpyHostToDevice));
//	checkCuda(cudaMemcpy(dev_chan, chan, sizeChannel, cudaMemcpyHostToDevice));
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\nthoi tian thuc hien copy bo nho: %f", milliseconds);
//
//	cudaEventRecord(start);
//	kernel << < gridSize, blockSize>> > (dev_chan, dev_gain, delt, count, iq_buff_size, dev_iq_buff);
//	cudaThreadSynchronize();
//
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds2 = 0;
//	cudaEventElapsedTime(&milliseconds2, start, stop);
//	printf("\nthoi tian thuc hien trong gpu: %f", milliseconds2);
//	checkCuda(cudaMemcpy(iq_buff, dev_iq_buff, sizeIq_buff, cudaMemcpyDeviceToHost));
//
//	checkCuda(cudaFree(dev_chan));
//	checkCuda(cudaFree(dev_gain));
//	checkCuda(cudaFree(dev_iq_buff));
//}