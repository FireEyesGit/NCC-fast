#include <opencv2/opencv.hpp>
#include "complex_operate.hpp"

using namespace cv;

//ԭͼ��С��640��480, ģ���С��116��110
//����Ĺ�һ�����ƥ���㷨��û���õ��κ������ֶΣ���ncc��������ʱ���Լ��23������
//ֻ��ģ���ƽ�����û���ͼ���٣�ʱ����ٵ�15��
//��ģ���ԭͼ����ƽ�����û���ͼ���٣�ʱ����ٵ�9.5��
//��ϻ���ͼ�����ٸ���Ҷ���㻥��أ�ʱ����ٵ�0.95������

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

	//�û���ͼ���ټ���ģ���ԭͼ��ƽ����
	Mat t_sum, t_sqsum,src_sum,src_sqsum;
	integral(templ,t_sum,t_sqsum, CV_64FC1);
	integral(src, src_sum, src_sqsum, CV_64FC1);
	double* ptr = t_sqsum.ptr<double>(templ.rows);
	double sum_templ = ptr[templ.cols];                                        //ֻ�Լ���ģ��ƽ���ͽ��л���ͼ���٣�ʱ����ٵ�15��

	double num = 0., den = 0.;
	double q0, q1, q2, q3;
	
	//��fft���ټ��㻥���
	Mat src_complex = fft(src, false);
	Mat templ_1;
	copyMakeBorder(templ, templ_1, 0, rows - 1, 0, cols - 1, BORDER_CONSTANT, 0.);   //��ģ������������ԭͼ��Сһ��
	Mat templ_complex = fft(templ_1, false);
	Mat conj_templ = conjugate(templ_complex);
	Mat R = real(fft(complexMultiplication(src_complex, conj_templ),true));          //��ϻ���ͼ�����ٸ���Ҷ���㻥��أ�ʱ����ٵ�0.95������

	for (int i = 0; i < result.rows; i++)
	{
		float* result_ptr = result.ptr<float>(i);                       //��Ϊresult��������CV_32FC1���������Ӧ��ָ����ָ�������Ϊfloat����uchar��double�������
		double* p1 = src_sqsum.ptr<double>(i + templ.rows);
		double* p2 = src_sqsum.ptr<double>(i);
		
		for (int j = 0; j < result.cols; j++)
		{
			num = R.at<float>(i, j);
			q0 = p1[j + templ.cols];                                     //��ָ�����ڷ���Ԫ�ر���.at<>()������΢��Щ����������
			q1 = p2[j + templ.cols];
			q2= p1[j];
			q3= p2[j];
			den = q0 - q1 - q2 + q3;                                     //����ģ�����ͼ��ԭͼ�����ͼ���٣����Ի�������٣�ʱ����ٵ�9.5��
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

	normalize(resultMap, resultMap, 0, 1,NORM_MINMAX);                 //һ���һ������NORM_MINMAX������ÿ��Ԫ��������һ����Χ�ڣ����������Ĭ�ϵ�ģʽ�ǹ�һ��������
	imshow("bb", resultMap);
	time = ((double)getTickCount() - time) / getTickFrequency();
	std::cout << time;                                                            //ֻ��¼ncc�㷨�����ѵ�ʱ�䣬�������õ�ģ���ԭͼ����ʱ23������
	

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(resultMap, &minVal, &maxVal, &minPos, &maxPos);
	std::cout << maxPos;

	rectangle(srcImage, maxPos, Point(maxPos.x + templImage.cols, maxPos.y + templImage.rows),Scalar(0,0,255));
	imshow("aa", srcImage);
	waitKey(0);
}