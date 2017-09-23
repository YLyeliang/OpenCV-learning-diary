#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/video/video.hpp"  

#include <iostream>  
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	/************************************************************************/
	/* 1.  Using the cv::RNG random number generator:
	a.  Generate and print three floating-point numbers, each drawn from a uniform
	distribution from 0.0 to 1.0.
	b.  Generate  and  print  three  double-precision  numbers,  each  drawn  from  a
	Gaussian distribution centered at 0.0 and with a standard deviation of 1.0.
	c.  Generate and print three unsigned bytes, each drawn from a uniform distri‐
	bution from 0 to 255.
	/************************************************************************/
	RNG rng = theRNG();
	//a
	float f1 = rng.uniform(0.0f, 1.0f);
	float f2 = rng.uniform(0.0f, 1.0f);
	float f3 = rng.uniform(0.0f, 1.0f);
	//cout << "first floating-point number:" << f1 << endl;
	//cout << "second floating-point number:" << f2 << endl;
	//cout << "third floating-point number:" << f3 << endl;
	////b
	//Vec3d vec3d;
	//rng.fill(vec3d, RNG::NORMAL, 0., 1.);
	//cout << " d1 " << vec3d[0] << " d2 " << vec3d[1] << " d3 " << vec3d[2] << endl;
	////c
	//unsigned bytes1 = rng.uniform(0, 255);
	//unsigned bytes2 = rng.uniform(0, 255);
	//unsigned bytes3 = rng.uniform(0, 255);
	//cout << "bytes1 " << bytes1 << "bytes2 " << bytes2 << "bytes3 " << bytes3 << endl;

	///************************************************************************/
	///* 2.  Using  the  fill()  method  of  the  cv::RNG  random  number  generator,  create  an
	//array of:
	//a.  20 floating-point numbers with a uniform distribution from 0.0 to 1.0.
	//b.  20  floating-point  numbers  with  a  Gaussian  distribution  centered  at  0.0  and
	//with a standard deviation of 1.0.
	//c.  20 unsigned bytes with a uniform distribution from 0 to 255.
	//d.  20 color triplets, each of three bytes with a uniform distribution from 0 to 255.                                                                     */
	///************************************************************************/
	////a
	//Mat matfloat20(20, 1, CV_32FC1, Scalar(0));
	//rng.fill(matfloat20, RNG::UNIFORM, 0.0f, 1.0f);
	//cout << "matfloat20 array" << matfloat20 << endl;
	////b
	//Mat matfloat20g(20, 1, CV_32FC1, Scalar(0));
	//rng.fill(matfloat20g, RNG::NORMAL, 0.0, 1.0);
	//cout << "Gaussian distribution number" << matfloat20g << endl;
	////c
	//Mat matunsigned20(20, 1, CV_8UC1, Scalar(0));
	//rng.fill(matunsigned20, RNG::UNIFORM, 0, 255);
	//cout << "unsined " << matunsigned20 << endl;
	////d
	//Mat matColor20 = Mat(20, 1, CV_8UC3, Scalar(0));
	//rng.fill(matColor20, RNG::UNIFORM, 0, 255);
	//cout << "color triplets" << matColor20 << endl;

	/************************************************************************/
	/* 3.  Using the cv::RNG random number generator, create an array of 100 three-byte
	objects such that:
	a.  The first and second dimensions have a Gaussian distribution, centered at 64
	and 192, respectively, each with a variance of 10.
	b.  The third dimension has a Gaussian distribution, centered at 128 and with a
	variance of 2.
	c.  Using the cv::PCA object, compute a projection for which maxComponents=2.
	d.  Compute the mean in both dimensions of the projection; explain the result.                                                                     */
	/************************************************************************/
	Mat matInt100 = Mat(100, 1, CV_32FC3, Scalar(0));
	//a
	vector<Mat> planes;
	split(matInt100, planes);
	rng.fill(planes[0], RNG::NORMAL, 64, 10);
	rng.fill(planes[1], RNG::NORMAL, 192, 10);
	//cout << "centered at 64:\n" << planes[0] << "\n" << "centerd at 192:\n" << planes[1] << endl;
	//b
	rng.fill(planes[2], RNG::NORMAL, 128, 2);
	//cout << "centered at 128:\n" << planes[2] << endl;
	//c
	PCA pca(planes[0], Mat(), CV_PCA_DATA_AS_ROW, 2);
	planes[0] = pca.project(planes[0]);
	pca(planes[1], Mat(), CV_PCA_DATA_AS_ROW, 2);
	planes[1] = pca.project(planes[1]);
	pca(planes[2], Mat(), CV_PCA_DATA_AS_ROW, 2);
	planes[2] = pca.project(planes[2]);
	cout << "projection of planes1:\n" << planes[0] << "\n" << "projection of planes2:\n" << planes[1] << "\n" << "projection of planes3:\n"
		<< planes[2] << endl;
	//d
	f1 = 0;
	f2 = 0;
	f3 = 0;
	for (int i = 0;i < 100;i++)
	{
		f1 += planes[0].at<float>(i, 0);
		f2 += planes[1].at<float>(i, 0);
		f3 += planes[2].at<float>(i, 0);
	}
	f1 = f1 / 100;
	f2 = f2 / 100;
	f3 = f3 / 100;
	cout << "f1" << f1 << "f2 " << f2 << "f3 " << f3 << endl;
	//4
	Matx32d AX(1, 1,
		0, 1,
		-1, 1);
	Mat A = static_cast<Mat>(AX);
	Mat U, W, V;
	SVD::compute(A, W, U, V);
	cout << W <<"\n"<< U <<"\n"<< V;
	getchar();
	return 0;

}




//opencv中的阵列操作函数
//int main() {
//	Mat src(Size(5,5), CV_32FC1);
//	Mat b;
//	Mat c;
//	int sum;
//	//double c;
//	int i = 1;
//	for (int x = 0;x < 5;++x) {
//		for (int y = 0;y < 5;++y) {
//			src.at<float>(y, x) = i;
//			i++;
//		}
//	}
//	cout << src << endl;
//
//	//convertScaleAbs(src, b, 2, 1);
//	//invert(src, c,DECOMP_LU);
//	//log(src, c);
//	meanStdDev(src, b, c);
//	cout << b << endl;
//	cout << c << endl;
//	//cout << mean(src) << endl;
//	//completeSymm(a,true);
//	//cout << b << endl;
//	system("pause");
//}

//addWeighted()
//int main() {
//	Mat src1 = imread("D://1.jpg");
//	Mat src2 = imread("D://3.jpg");
//	int x = 50;
//	int y = 100;
//	double alpha = 0.4;
//	double beta = 0.6;
//	Mat roi1(src1, Rect(x, y, 400,400 ));//设置src1的ROI(region of interest) 宽、高可由.cols .rows得到
//	Mat roi2(src2, Rect(0, 0, 400, 400));//设置src2的ROI
//	imshow("roi1", roi1);
//	imshow("roi2", roi2);
//	addWeighted(roi1, alpha, roi2, beta, 0.0, roi2);//(进行线性加权求和操作,将结果输入到src2的ROI中
//	imshow("photo", src2);//显示ROI中图像修改后的整个src2图像
//	waitKey(0);
//	return 0;
//}




//opticalflow
//static VideoCapture cap;
//static  unsigned int frame_count = 0;
//
//void compute_absolute_mat(const Mat& in, Mat & out);

//int main()
//{
//	cap.open("D://input.avi");
//
//	if (!cap.isOpened()) {
//		cout << "视频读取失败！" << endl;
//		return -1;
//	}
//	VideoWriter writer;
//	double fps = cap.get(CV_CAP_PROP_FPS);
//	Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
//	writer.open("D://Detection1.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, size);//存储路径
//	Mat img, gray, prvGray, optFlow, absoluteFlow, img_for_show,dst;
//	while (1) {
//		cap >> img;
//		if (img.empty()) break;
//
//		cvtColor(img, gray, CV_BGR2GRAY);
//		if (prvGray.data) {
//			calcOpticalFlowFarneback(prvGray, gray, optFlow, 0.5, 3, 15, 3, 5, 1.2, 0); //使用论文参数      
//			compute_absolute_mat(optFlow, absoluteFlow);
//			normalize(absoluteFlow, img_for_show, 0, 255, NORM_MINMAX, CV_8UC1);
//
//			imshow("opticalFlow", img_for_show);
//			imshow("resource", img);
//			cvtColor(img_for_show, dst, CV_GRAY2BGR);
//			//writer << dst;
//		}
//		swap(prvGray, gray);
//
//		waitKey(1);
//	}
//
//	return 0;
//}
//
//void compute_absolute_mat(const Mat& in, Mat & out)
//{
//	if (out.empty()) {
//		out.create(in.size(), CV_32FC1);
//	}
//
//	const Mat_<Vec2f> _in = in;
//	//遍历吧，少年  
//	for (int i = 0; i < in.rows; ++i) {
//		float *data = out.ptr<float>(i);
//		for (int j = 0; j < in.cols; ++j) {
//			double s = _in(i, j)[0] * _in(i, j)[0] + _in(i, j)[1] * _in(i, j)[1];
//			if (s>1) {
//				data[j] = std::sqrt(s);
//			}
//			else {
//				data[j] = 0.0;
//			}
//
//		}
//	}
//}

//int main() {
//
//	//create a 10x10 sparse matrix with a few nonzero elements
//	int size[] = { 10,10 };
//	SparseMat sm(2, size, CV_32F);
//
//	for (int i = 0;i < 10;i++) {			//fill the array
//		int idx[2];
//		idx[0] = size[0] * rand();
//		idx[1] = size[1] * rand();
//
//		sm.ref<float>(idx) += 1.0f;
//	}
//
//	//print out the nonzero elements
//	print_matrix(sm);
//	//SparseMatConstIterator_<float> it = sm.begin<float>();
//	//SparseMatConstIterator_<float> it_end = sm.end<float>();
//
//	//for (;it != it_end;++it) {
//	//	const SparseMat::Node* node = it.node();
//	//	printf(" (%3d,%3d) %f\n", node->idx[0], node->idx[1], *it);
//	//}
//	for (;;)
//		waitKey(0);
//}


//int main() {
//	Mat frame, gray, mask;
//	VideoCapture capture("D://input.avi");
//	VideoWriter writer;
//	double fps = capture.get(CV_CAP_PROP_FPS);
//	Size size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
//	writer.open("D://Detection1.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
//	//capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
//	if (!capture.isOpened())
//	{
//		cout << "No camera or video input!\n" << endl;
//		return -1;
//	}
//
//	ViBe_BGS Vibe_Bgs;
//	bool count = true;
//
//	while (1)
//	{
//		capture >> frame;
//		if (frame.empty())
//			continue;
//
//		cvtColor(frame, gray, CV_RGB2GRAY);
//		if (count)
//		{
//			Vibe_Bgs.init(gray);
//			Vibe_Bgs.processFirstFrame(gray);
//			cout << " Training ViBe complete!" << endl;
//			count = false;
//		}
//		else
//		{
//			Vibe_Bgs.testAndUpdate(gray);
//			mask = Vibe_Bgs.getMask();
//			morphologyEx(mask, mask, MORPH_OPEN, Mat());
//			imshow("mask", mask);
//			cvtColor(mask, mask, CV_GRAY2RGB);
//			writer << mask;
//		}
//
//		imshow("input", frame);
//		
//		if (cvWaitKey(10) == 27)
//			break;
//	}
//
//	return 0;
//}

//
//int main() 
//	{
//		VideoCapture videoCap("D://office.avi");
//		if (!videoCap.isOpened())
//		{
//			return -1;
//		}
//		double videoFPS = videoCap.get(CV_CAP_PROP_FPS);  //获取帧率
//		Size size((int)videoCap.get(CAP_PROP_FRAME_WIDTH), (int)videoCap.get(CAP_PROP_FRAME_HEIGHT));
//		/*double videoPause = 1000ideoFPS;*/
//		Mat framePrePre; //上上一帧  
//		Mat framePre; //上一帧  
//		Mat frameNow; //当前帧  
//		Mat frameDet; //运动物体  
//		VideoWriter writer;
//		writer.open("D://Detection.avi", CV_FOURCC('M', 'J', 'P', 'G'),videoFPS,size);
//		videoCap >> framePrePre;
//		videoCap >> framePre;
//		cvtColor(framePrePre, framePrePre, CV_RGB2GRAY);
//		cvtColor(framePre, framePre, CV_RGB2GRAY);
//		int save = 0;
//		for (;;)
//		{
//			videoCap >> frameNow;
//			if (frameNow.empty())
//			{
//				break;
//			}
//			cvtColor(frameNow, frameNow, CV_BGR2GRAY);
//			Mat Det1;
//			Mat Det2;
//			absdiff(framePrePre, framePre, Det1);  //帧差1  
//			absdiff(framePre, frameNow, Det2);     //帧差2  
//			threshold(Det1, Det1, 0, 255, CV_THRESH_OTSU);  //自适应阈值化  
//			threshold(Det2, Det2, 0, 255, CV_THRESH_OTSU);
//			Mat element = getStructuringElement(0, Size(3, 3));  //膨胀核  
//			dilate(Det1, Det1, element);    //膨胀  
//			dilate(Det2, Det2, element);
//			bitwise_and(Det1, Det2, frameDet);
//			framePrePre = framePre;
//			framePre = frameNow;
//			imshow("Video", frameNow);
//			imshow("Detection", frameDet);
//			cvtColor(frameDet, frameDet, CV_GRAY2BGR);
//			writer << frameDet;
//			waitKey(10);
//		}
//		videoCap.release();
//		return 0;
//	}



//int main() {
//	VideoCapture video("D://office.avi");
//	Mat dst;
//	Mat frame, mask, thresholdImage, output;
//	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));//腐蚀或膨胀的核大小
//	double videoFPS = video.get(CV_CAP_PROP_FPS);  //获取帧率
//	Size size((int)video.get(CAP_PROP_FRAME_WIDTH), (int)video.get(CAP_PROP_FRAME_HEIGHT));
//	VideoWriter writer;
//	writer.open("D://Detection.avi", CV_FOURCC('M', 'J', 'P', 'G'),videoFPS,size);
//	int frameNum = 0;//初始化帧数
//					 //video >> frame;
//	//BackgroundSubtractorMOG2 bgSubtractor(20, 16, false);
//	Ptr<BackgroundSubtractorMOG2>bgsubtractor=createBackgroundSubtractorMOG2();
//	bgsubtractor->setHistory(20);
//	bgsubtractor->setVarThreshold(16);
//	bgsubtractor->setDetectShadows(true);
//	while (true) {
//		video >> frame;//前进一帧
//		++frameNum;//帧数计算
//		namedWindow("frame", WINDOW_NORMAL);//显示当前帧图像
//		imshow("frame", frame);
//		bgsubtractor->apply(frame, mask, 0.001);//背景减除
//		erode(mask, mask, element);//膨胀
//		dilate(mask, mask, element);//腐蚀
//		namedWindow("mask", WINDOW_NORMAL);//分割后图像
//		imshow("mask", mask);
//		cvtColor(mask, dst, CV_GRAY2RGB);
//		writer << dst;
//		//if (frameNum == 525)
//		//{
//		//	namedWindow("image", WINDOW_NORMAL);
//		//	imshow("image", frame);
//		//	imwrite("E:\\毕业设计\\图片\\520.jpg", frame);
//		//	namedWindow("dst", WINDOW_NORMAL);
//		//	imshow("dst", mask);
//		//	imwrite("E:\\毕业设计\\图片\\520M.jpg", mask);
//		//}
//		//btractor.getBackgroundImage(output);//得到背景图
//		//namedWindow("bg", WINDOW_NORMAL);
//		//imshow("bg", output);
//		waitKey(10);//0表示手动逐帧进行；其他数字表示自动运行
//		if (frame.empty()) break;
//		if (mask.empty()) break;
//	}
//	return 0;
//}




//int main() {
//	const int n_mat_size = 5;
//	const int n_mat_sz[] = { n_mat_size,n_mat_size,n_mat_size };
//	Mat n_mat0(3, n_mat_sz, CV_32FC1);
//	Mat n_mat1(3, n_mat_sz, CV_32FC1);
//
//	RNG rng;
//	rng.fill(n_mat0, RNG::UNIFORM, 0.f, 1.f);
//	rng.fill(n_mat1, RNG::UNIFORM, 0.f, 1.f);
//
//	const Mat* arrays[] = { &n_mat0,&n_mat1,0 };
//	Mat my_planes[2];
//	NAryMatIterator it(arrays, my_planes);
//
//	float s = 0.f;
//	int n = 0;
//	for (int p = 0;p < it.nplanes;p++, ++it) {
//		s += sum(it.planes[0])[0];
//		s += sum(it.planes[1])[1];
//		n++;
//	}
//	cout << "s=" << s << "," << "n=" << n << endl;
//	waitKey(0);
//	return 0;
//}

//int main() {
//	namedWindow("video", WINDOW_AUTOSIZE);
//	namedWindow("Log_polar", WINDOW_AUTOSIZE);
//	VideoCapture cap("D://vtest.avi");
//	double fps = cap.get(CAP_PROP_FPS);
//	Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
//	VideoWriter writer;
//	writer.open("D://test.avi",CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
//	Mat logpolar_frame, bgr_frame;
//	for (;;) {
//		cap >> bgr_frame;
//		if (bgr_frame.empty()) break;
//		imshow("video", bgr_frame);
//		logPolar(bgr_frame, logpolar_frame, Point2f(bgr_frame.cols / 2, bgr_frame.rows / 2), 40, WARP_FILL_OUTLIERS);
//		imshow("Log_polar", logpolar_frame);
//		writer << logpolar_frame;
//		char c = waitKey(10);
//		if (c == 27) break;
//	}
//	cap.release();
//}



//int main() {
//	Mat img_rgb,img_gry,img_cny,img_pyr;
//	int x = 16, y = 32;
//	img_rgb = imread("D://lena.jpg");
//	cvtColor(img_rgb, img_gry, COLOR_BGR2GRAY);
//	imshow("gray",img_gry);
//	pyrDown(img_gry, img_pyr);
//	Canny(img_pyr, img_cny, 10, 100, 3, true);
//	imshow("canny", img_cny);
//	Vec3b intensity = img_rgb.at<Vec3b>(y, x);
//	uchar blue = intensity[0];
//	uchar green = intensity[1];
//	uchar red = intensity[2];
//	cout << "At(x,y)=(" << x << "," << y << "):(blue,green,red)=(" << (unsigned int)blue << "," << (unsigned int)green << ","
//		<< (unsigned int)red << ")" << endl;
//	cout << "Gray pixel there is:" << (unsigned int)img_gry.at<uchar>(y, x) << endl;
//	x /= 4;
//	y /= 4;
//	cout << "Pyramid2 pixel there is:" << (unsigned int)img_pyr.at<uchar>(y, x) << endl;
//	waitKey(0);
//}




/*
int g_slider_position = 0;//滚动条位置
int g_run = 1, g_dontset = 0;
VideoCapture g_cap;
void onTrackbarSlide(int pos, void *)
{
	g_cap.set(CAP_PROP_POS_FRAMES, pos);//将pos的值设为当前帧,改动滚动条的值返回该函数，并将值传递给FRAMES，同时改为single-step
	if (!g_dontset)
		g_run = 1;
	g_dontset=0;
}

int main() {
	namedWindow("video", WINDOW_AUTOSIZE);
	g_cap.open("D://vtest.avi");
	int frames = (int)g_cap.get(CAP_PROP_FRAME_COUNT);//总帧数
	int tmpw = (int)g_cap.get(CAP_PROP_FRAME_WIDTH);//帧宽度
	int tmph = (int)g_cap.get(CAP_PROP_FRAME_HEIGHT);//帧高度
	cout << "Video has " << frames << "frames of dimenstions(" << tmpw << "," << tmph << ".)" << endl;
	createTrackbar("Position", "video", &g_slider_position, frames, onTrackbarSlide);//滚动条名，窗口名，过程值，最大值，返回函数
	//createTracbar中的第五个参数返回函数中的函数定义中，第一个参数必须为当前帧数，即pos。第二个为user data,加*即可。
	Mat frame;
	
	for (;;) {
		if (g_run != 0) {
			g_cap >> frame;
			if (frame.empty()) break;
			int current_pos = (int)g_cap.get(CAP_PROP_POS_FRAMES);//得到当前帧数
			g_dontset = 1;
			setTrackbarPos("Position", "video", current_pos);
			imshow("video", frame);
			g_run -= 1;
		}
		char c = (char)waitKey(10);
		if(c=='s') //single step
		{
			g_run = 1;
			cout << "Single step,run=" << g_run << endl;
		}
		if (c == 'r') //run mode
		{
			g_run = -1;
			cout << "Run mode,run=" << g_run << endl;
		}
		if (c == 27) break;//Esc的ASCII码为十进制27
	}
	return (0);
}
*/