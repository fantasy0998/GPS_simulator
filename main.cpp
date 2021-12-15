#include "gpssim.h"
#include <stdio.h>
#include <complex>
#include "kernal.h"


int main()
{
	const char* rfile= "D:\\PL\\GR_GPS_Cuda_mix\\GPSL1_sim_quantify\\brdc2017_0660.17n";
	const char* tfile = "D:\\PL\\creat_so\\GPSL1_sim_so-v2\\tra.csv";
	FILE* fp = NULL,*fpw=NULL;
	fp = fopen("gpssim_.bin", "wb+");
	fpw = fopen("gpssim.bin", "wb+");

	typedef std::complex<float> complexf;
	int generated_samples = 0;
	int samp_freq = 5e6;
	int simu_time = 10; //调用一次函数的仿真时间
	int time_all = 400 ;  //总仿真时间
	int buff_size = (samp_freq * simu_time / 10);//需要的采样点数
	complexf *buff = new complexf[buff_size];
	int* qua_buff = (int*)malloc(sizeof(int) * buff_size / 5);
	int ibit = 0;
	transfer_parameter tp;
	gpstime_t g0;
	tp.g0.week = -1;
	g0.sec = 10.0;
	tp.xyz = (double(*)[3]) malloc(sizeof(double) * time_all * 3);
	tp.navbit = (char*)malloc(sizeof(char) * MAX_SAT * 1800);//一个子帧300个字
	tp.neph = readRinexNavAll(tp.eph, &tp.ionoutc, rfile);
	readUserMotion(tp.xyz, tfile, time_all);


	Table GPSL1table;
	int* CAcode;
	CAcode = (int*)malloc(sizeof(int) * MAX_SAT * 1023);
	for (int i = 0; i < MAX_SAT; i++)
	{
		codegen((CAcode + i * 1023), i + 1);
	}
	GPSL1table.CAcode = CAcode;
	GPSL1table.i_buff = (float*)malloc(2 * sizeof(float) * buff_size);
	float *dev_i_buff=NULL;
	//checkCuda(cudaHostAlloc((void**)&dev_i_buff, buff_size * sizeof(float) * 2, cudaHostAllocDefault));
	for (int i = 0; i < (int)(time_all / simu_time); i++)
	{
		memset(qua_buff, 0, sizeof(int) * buff_size / 5);
		generated_samples = gps_sim(&GPSL1table,&tp, buff, buff_size, samp_freq, 0, simu_time,dev_i_buff);//需要清理一次complex数组
		for (int isamp = 0; isamp < buff_size; isamp++)
		{
			int bcount = isamp / 10;
			int rcount = isamp % 10;
			if (rcount < 5)
			{
				ibit = 0;
				ibit = quantify(buff[isamp].real(), 1);
				qua_buff[bcount * 2 + 1] |= ibit << (6 * (5 - rcount) - 3);//存储i路采样点
				ibit = 0;
				ibit = quantify(buff[isamp].imag(), 1);
				qua_buff[bcount * 2 + 1] |= ibit << (6 * (5 - rcount) - 4);//存储q路采样点
			}
			else
			{
				rcount -= 5;
				ibit = 0;
				ibit = quantify(buff[isamp].real(), 1);
				qua_buff[bcount * 2] |= ibit << (6 * (5 - rcount) - 1);//存储i路采样点
				ibit = 0;
				ibit = quantify(buff[isamp].imag(), 1);
				qua_buff[bcount * 2] |= ibit << (6 * (5 - rcount) - 2);//存储q路采样点
			}
		}

		//fwrite(qua_buff, 4, buff_size / 5, fp);
		//fwrite(buff, 8, buff_size, fpw);
		if (i == 0)
		{
			//fwrite(buff, 8, (int)(buff_size*0.9), fpw);
			fwrite(qua_buff, 4, buff_size*0.9 / 5, fp);
		}
		else
		{
			//fwrite(buff, 8, buff_size, fpw);
			fwrite(qua_buff, 4, buff_size / 5, fp);
		}
		//if (i == 1)
		//{
		//	for (int j =5e5; j < 5e5+100; j++)
		//	{
		//		printf("%.5f %.5f\n", buff[j].real(), buff[j].imag());
		//	}
		//	printf("\n\n");
		//}
	}

	delete[]buff;
	fclose(fpw);
	fclose(fp);
	free(qua_buff);
	free(tp.navbit);
	free(tp.xyz);
	int aa=getchar();
}