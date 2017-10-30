#include <opencv2/opencv.hpp>
#include "complex_operate.hpp"

using namespace cv;

//原图大小：640×480, 模板大小：116×110
//纯粹的归一化相关匹配算法，没有用到任何提速手段，纯ncc函数运行时间大约在23秒左右
//只对模板的平方和用积分图加速，时间减少到15秒
//对模板和原图计算平方和用积分图加速，时间减少到9.5秒
//结合积分图及快速傅里叶计算互相关，时间减少到0.95秒左右

void ncc(Mat& srcImage, Mat& templImage, Mat& result)
{
	int rows = srcImage.rows - templImage.rows + 1;
	assert(templImage.rows <= srcImage.rows);
	int cols = srcImage.cols - templImage.cols + 1;
	assert(templImage.cols <= srcImage.cols);
	result.create(rows, cols, CV_32FC1);

	Mat src, templ;
	cvtColor(srcImage, src, COLOR_BGR2GRAY);
	cvtColor(templImage, templ, COLOR_BGR2GRAY);

	//用积分图加速计算模板和原图的平方和
	Mat t_sum, t_sqsum,src_sum,src_sqsum;
	integral(templ,t_sum,t_sqsum, CV_64FC1);
	integral(src, src_sum, src_sqsum, CV_64FC1);
	double* ptr = t_sqsum.ptr<double>(templ.rows);
	double sum_templ = ptr[templ.cols];                                        //只对计算模板平方和进行积分图加速，时间减少到15秒

	double num = 0., den = 0.;
	double q0, q1, q2, q3;
	
	//用fft快速计算互相关
	Mat src_complex = fft(src, false);
	Mat templ_1;
	copyMakeBorder(templ, templ_1, 0, rows - 1, 0, cols - 1, BORDER_CONSTANT, 0.);   //将模板零延拓至与原图大小一致
	Mat templ_complex = fft(templ_1, false);
	Mat conj_templ = conjugate(templ_complex);
	Mat R = real(fft(complexMultiplication(src_complex, conj_templ),true));          //结合积分图及快速傅里叶计算互相关，时间减少到0.95秒左右

	for (int i = 0; i < result.rows; i++)
	{
		float* result_ptr = result.ptr<float>(i);                       //因为result的类型是CV_32FC1，所以其对应的指针所指向的类型为float，用uchar和double都会出错
		double* p1 = src_sqsum.ptr<double>(i + templ.rows);
		double* p2 = src_sqsum.ptr<double>(i);
		
		for (int j = 0; j < result.cols; j++)
		{
			num = R.at<float>(i, j);
			q0 = p1[j + templ.cols];                                     //用指针用于访问元素比用.at<>()访问略微快些，但不明显
			q1 = p2[j + templ.cols];
			q2= p1[j];
			q3= p2[j];
			den = q0 - q1 - q2 + q3;                                     //加上模板积分图和原图像积分图提速，不对互相关提速，时间减少到9.5秒
			result_ptr[j] = (float)(num / (sqrt(sum_templ)*sqrt(den)));
			num = 0.;
			den = 0.;
		}
	}
}

int main()
{
	Mat srcImage = imread("C:\\Users\\LIUU\\Pictures\\src.jpg");
	Mat templImage = imread("C:\\Users\\LIUU\\Pictures\\roi_1.jpg");

	double time = static_cast<double>(getTickCount());
	Mat resultMap;
	ncc(srcImage, templImage, resultMap);

	normalize(resultMap, resultMap, 0, 1,NORM_MINMAX);                 //一般归一化常用NORM_MINMAX，即将每个元素限制在一定范围内，但这个函数默认的模式是归一化二范数
	imshow("bb", resultMap);
	time = ((double)getTickCount() - time) / getTickFrequency();
	std::cout << time;                                                            //只记录ncc算法所花费的时间，对于我用的模板和原图，耗时23秒左右
	

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(resultMap, &minVal, &maxVal, &minPos, &maxPos);
	std::cout << maxPos;

	rectangle(srcImage, maxPos, Point(maxPos.x + templImage.cols, maxPos.y + templImage.rows),Scalar(0,0,255));
	imshow("aa", srcImage);
	waitKey(0);
}