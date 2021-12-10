#include "orbDetector.h"

namespace PoseTracking
{

//�������ӣ�һ�ȶ�Ӧ�Ŷ��ٻ���
const float factorPI = (float)(CV_PI / 180.f);

//�������Ԥ�ȶ���õ�����㼯��256��ָ������ȡ��256bit����������Ϣ��ÿ��bit��һ�Ե�Ƚϵ�����4=2*2��ǰ���2����Ҫ�����㣨һ�Ե㣩���бȽϣ������2��һ��������������
//���ֱ�ʾ��������������������������ƫ����
static int bit_pattern_31_[256 * 4] =
{
	8,-3, 9,5/*mean (0), correlation (0)*/,				//����ľ�ֵ�������û�п�����ʲô��˼
	4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
	-11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
	7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
	2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
	1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
	-2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
	-13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
	-13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
	10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
	-13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
	-11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
	7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
	-4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
	-13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
	-9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
	12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
	-3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
	-6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
	11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
	4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
	5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
	3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
	-8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
	-2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
	-13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
	-7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
	-4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
	-10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
	5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
	5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
	1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
	9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
	4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
	2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
	-4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
	-8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
	4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
	0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
	-13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
	-3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
	-6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
	8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
	0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
	7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
	-13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
	10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
	-6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
	10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
	-13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
	-13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
	3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
	5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
	-1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
	3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
	2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
	-13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
	-13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
	-13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
	-7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
	6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
	-9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
	-2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
	-12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
	3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
	-7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
	-3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
	2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
	-11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
	-1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
	5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
	-4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
	-9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
	-12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
	10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
	7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
	-7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
	-4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
	7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
	-7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
	-13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
	-3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
	7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
	-13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
	1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
	2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
	-4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
	-1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
	7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
	1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
	9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
	-1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
	-13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
	7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
	12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
	6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
	5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
	2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
	3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
	2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
	9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
	-8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
	-11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
	1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
	6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
	2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
	6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
	3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
	7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
	-11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
	-10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
	-5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
	-10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
	8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
	4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
	-10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
	4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
	-2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
	-5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
	7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
	-9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
	-5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
	8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
	-9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
	1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
	7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
	-2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
	11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
	-12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
	3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
	5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
	0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
	-9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
	0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
	-1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
	5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
	3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
	-13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
	-5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
	-4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
	6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
	-7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
	-13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
	1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
	4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
	-2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
	2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
	-2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
	4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
	-6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
	-3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
	7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
	4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
	-13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
	7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
	7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
	-7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
	-8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
	-13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
	2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
	10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
	-6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
	8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
	2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
	-11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
	-12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
	-11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
	5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
	-2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
	-1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
	-13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
	-10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
	-3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
	2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
	-9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
	-4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
	-4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
	-6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
	6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
	-13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
	11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
	7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
	-1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
	-4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
	-7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
	-13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
	-7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
	-8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
	-5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
	-13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
	1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
	1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
	9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
	5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
	-1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
	-9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
	-1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
	-13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
	8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
	2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
	7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
	-10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
	-10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
	4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
	3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
	-4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
	5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
	4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
	-9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
	0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
	-12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
	3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
	-10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
	8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
	-8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
	2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
	10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
	6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
	-7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
	-3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
	-1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
	-3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
	-8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
	4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
	2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
	6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
	3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
	11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
	-3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
	4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
	2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
	-10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
	-13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
	-13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
	6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
	0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
	-13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
	-9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
	-13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
	5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
	2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
	-1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
	9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
	11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
	3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
	-1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
	3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
	-13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
	5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
	8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
	7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
	-10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
	7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
	9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
	7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
	-1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

/**
* @brief ���캯��
* @detials ֮���Ի���������Ӧֵ����ֵ��ԭ���ǣ�������ʹ�ó�ʼ��Ĭ��FAST��Ӧֵ��ֵ��ȡͼ��cell�е������㣻�����ȡ����
* ��������Ŀ���㣬��ô�ͽ���Ҫ��ʹ�ý�СFAST��Ӧֵ��ֵ�����ٴ���ȡ���Ի�þ����ܶ��FAST�ǵ㡣
* @param[in] nfeatures         ָ��Ҫ��ȡ��������������Ŀ
* @param[in] scaleFactor       ͼ�������������ϵ��
* @param[in] nlevels           ָ����Ҫ��ȡ�������ͼ���������
* @param[in] iniThFAST         ��ʼ��Ĭ��FAST��Ӧֵ��ֵ
* @param[in] minThFAST         ��С��FAST��Ӧֵ��ֵ
*/
orbDetector::orbDetector(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST) :
	mnfeatures(_nfeatures), mscaleFactor(_scaleFactor), mnlevels(_nlevels),
	miniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
	//�洢ÿ��ͼ������ϵ����vector����Ϊ����ͼ����Ŀ�Ĵ�С
	mvScaleFactor.resize(mnlevels);
	//�洢���sigma^2����ʵ����ÿ��ͼ����Գ�ʼͼ���������ӵ�ƽ��
	mvLevelSigma2.resize(mnlevels);
	//���ڳ�ʼͼ����������������1
	mvScaleFactor[0] = 1.0f;
	mvLevelSigma2[0] = 1.0f;
	//Ȼ��������ͼ���������ͼ���൱�ڳ�ʼͼ�������ϵ�� 
	for (size_t i = 1; i < mnlevels; i++)
	{
		//��ʵ���������۳˼���ó�����
		mvScaleFactor[i] = mvScaleFactor[i - 1] * mscaleFactor;
		//ԭ�������sigma^2����ÿ��ͼ������ڳ�ʼͼ���������ӵ�ƽ��
		mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
	}

	//������������������������Ĳ����ĵ���
	mvInvScaleFactor.resize(mnlevels);
	mvInvLevelSigma2.resize(mnlevels);
	for (size_t i = 0; i < mnlevels; i++)
	{
		mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
		mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
	}

	//����ͼ�������vector��ʹ��������趨��ͼ�����
	mvImagePyramid.resize(mnlevels);

	//ÿ����Ҫ��ȡ������������������������ҲҪ����ͼ��������趨�Ĳ������е���
	mv_nFeaturesPerLevel.resize(mnlevels);

	//ͼƬ����������ϵ���ĵ���
	float factor = 1.0f / mscaleFactor;
	//��0��ͼ��Ӧ�÷��������������
	float nDesiredFeaturesPerScale = mnfeatures*(1 - factor) / (1 - (float)std::pow((double)factor, (double)mnlevels));

	//�������������������ģ���������ۼƼ������
	int sumFeatures = 0;
	//��ʼ������Ҫ��������������������ͼ����⣨��ѭ�����棩
	for (size_t level = 0; level < mnlevels - 1; level++)
	{
		//���� cvRound:��������
		mv_nFeaturesPerLevel[level] = std::round(nDesiredFeaturesPerScale);
		//�ۼ�
		sumFeatures += mv_nFeaturesPerLevel[level];
		//��ϵ��
		nDesiredFeaturesPerScale *= factor;
	}
	//����ǰ������������ȡ�����������ܻᵼ��ʣ��һЩ���������û�б����䣬��������ͽ�������������������䵽��ߵ�ͼ����
	mv_nFeaturesPerLevel[mnlevels - 1] = std::max(mnfeatures - sumFeatures, 0);

	//��Ա����pattern�ĳ��ȣ�Ҳ���ǵ�ĸ����������512��ʾ512���㣨������������Ǵ洢������������256*2*2��
	const int npoints = 512;

	//��ȡ���ڼ���BRIEF�����ӵ����������㼯ͷָ��
	//ע�⵽pattern0��������ΪPoints*,bit_pattern_31_��int[]�ͣ�����������Ҫ����ǿ������ת��
	const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;

	//ʹ��std::back_inserter��Ŀ���ǿ��Կ츲�ǵ��������pattern֮ǰ������
	//��ʵ����Ĳ������ǣ�����ȫ�ֱ�������ġ�int��ʽ�������������cv::point��ʽ���Ƶ���ǰ������еĳ�Ա������
	std::copy(pattern0, pattern0 + 512, std::back_inserter(mvpattern));

	//����������Ǻ����������ת�����йص�
	//Ԥ�ȼ���Բ��patch���еĽ���λ��
	//+1�е�1��ʾ�Ǹ�Բ���м���
	mv_umax.resize(HALF_PATCH_SIZE + 1);

	//����Բ������кţ�+1Ӧ���ǰ��м���Ҳ�����ǽ�ȥ��
	//NOTICE ע�����������к�ָ���Ǽ����ʱ�������кţ����еĺ�Բ�Ľǵ���45��Բ�Ľǵ�һ���ϣ�֮��������ѡ��
	//����ΪԲ���ϵĶԳ�����
	int vmax = std::floor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
	int vmin = std::ceil(HALF_PATCH_SIZE * sqrt(2.f) / 2);

	//�뾶��ƽ��
	const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;

	//����Բ�ķ��̼���ÿ�����ص�u����߽磨max��
	for (size_t v = 0; v <= HALF_PATCH_SIZE; ++v)
		mv_umax[v] = std::round(sqrt(hp2 - v * v));		//������Ǵ���0�Ľ������ʾx��������һ�еı߽�

	//������ʵ��ʹ���˶ԳƵķ�ʽ�������ķ�֮һ��Բ���ϵ�umax��Ŀ��Ҳ��Ϊ�˱����ϸ�ĶԳƣ�������ճ�����뷨��������cvRound�ͻ�����׳��ֲ��ԳƵ������
	//ͬʱ��Щ��������������㼯Ҳ���ܹ�������ת֮��Ĳ����������ˣ�
	for (size_t v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
	{
		while (mv_umax[v0] == mv_umax[v0 + 1])
			++v0;
		mv_umax[v] = v0;
		++v0;
	}
}

/**
* @brief �÷º������������������������������ͼ��������
*
* @param[in] _image                    ����ԭʼͼ��ͼ��
* @param[in] _mask                     ��Ĥmask
* @param[in & out] _keypoints                �洢������ؼ��������
* @param[in & out] _descriptors              �洢�����������ӵľ���
*/
void orbDetector::operator()(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	cv::Mat image = image_.clone();
	if (image.channels() != 1) cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// Step 2 ����ͼ�������
	ComputePyramid(image);

	// Step 3 ����ͼ��������㣬���ҽ���������о��Ȼ������ȵ�������������λ�˼��㾫��
	// �洢���е������㣬ע��˴�Ϊ��ά��vector����һά�洢���ǽ������Ĳ������ڶ�ά�洢������һ�������ͼ������ȡ������������
	std::vector < std::vector<cv::KeyPoint> > allKeypoints;
	//ʹ���Ĳ����ķ�ʽ����ÿ��ͼ��������㲢���з���
	ComputeKeyPointsOctTree(allKeypoints);

	// Step 4 ����ͼ�������ӵ��µľ���descriptors
	//ͳ������ͼ��������е�������
	int nkeypoints = 0;
	//��ʼ����ÿ��ͼ��������������ۼ�ÿ������������
	for (int level = 0; level < mnlevels; ++level)
		nkeypoints += (int)allKeypoints[level].size();

	//�����ͼ���������û���κε�������
	if (nkeypoints == 0) return;
	else
	{
		//���ͼ����������������㣬��ô�ʹ�������洢�����ӵľ���ע����������Ǵ洢����ͼ���������������������ӵ�
		descriptors.create(nkeypoints,		//�������������ӦΪ��������ܸ���
			32, 			//�������������ӦΪʹ��32*8=256λ������
			CV_8U);			//����Ԫ�صĸ�ʽ
	}

	//�������������������ȡ�����vector����
	keypoints.clear();
	//��Ԥ������ȷ��С�Ŀռ�
	keypoints.reserve(nkeypoints);

	//��Ϊ������һ��һ����еģ������������Ǹ������Ǵ洢����ͼ���������������������ӣ�����������������Offset���������桰Ѱַ��ʱ��ƫ������
	//������������������mat�еĶ�λ
	int offset = 0;
	//��ʼ����ÿһ��ͼ��
	for (int level = 0; level < mnlevels; ++level)
	{
		//��ȡ��allKeypoints�е�ǰ�������������ľ��
		std::vector<cv::KeyPoint> keypoints_ = allKeypoints[level];
		//�������������
		int nkeypointsLevel = (int)keypoints_.size();

		//�����������ĿΪ0����������ѭ����������һ�������
		if (nkeypointsLevel == 0)
			continue;

		// desc�洢��ǰͼ���������
		cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

		// Step 6 �����˹ģ����ͼ���������
		computeDescriptors(mvImagePyramid[level], 	//��˹ģ��֮���ͼ��ͼ��
			keypoints_,         	    //��ǰͼ���е������㼯��
			desc, 		            //�洢����֮���������
			mvpattern);	            //�������ģ��
								
		offset += nkeypointsLevel;  // ����ƫ������ֵ 

		// Step 6 �Էǵ�0��ͼ���е������������ָ�����0��ͼ��ԭͼ�񣩵�����ϵ��
		// �õ����в��������ڵ�0���������ŵ�_keypoints����
		// ���ڵ�0���ͼ�������㣬���ǵ�����Ͳ���Ҫ�ٽ��лָ���
		if (level != 0)
		{
			// ��ȡ��ǰͼ���ϵ�����ϵ��
			float scale = mvScaleFactor[level];
			// �����������е�������
			// �����㱾��ֱ�ӳ����ű����Ϳ�����
			for (auto& keypoint: keypoints_)
				keypoint.pt *= scale;
		}
		keypoints.insert(keypoints.end(), keypoints_.begin(), keypoints_.end());
	}
}

/**
* @brief ����ĳ�������ͼ�����������������
*
* @param[in] image                 ĳ�������ͼ��
* @param[in] keypoints             ������vector����
* @param[out] descriptors          ������
* @param[in] pattern               ����������ʹ�õĹ̶�����㼯
*/
void orbDetector::computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const std::vector<cv::Point>& pattern)
{
	//��ձ�����������Ϣ������
	descriptors = cv::Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

	//��ʼ����������
	for (size_t i = 0; i < keypoints.size(); i++)
		//��������������������
		computeOrbDescriptor(keypoints[i], 				//Ҫ���������ӵ�������
			image, 					      //�Լ���ͼ��
			&pattern[0], 				  //����㼯���׵�ַ
			descriptors.ptr((int)i));	  //��ȡ�����������ӵı���λ��
}

/**
* @brief ����ORB������������ӡ�ע�������ȫ�ֵľ�̬������ֻ�����ڱ��ļ��ڱ�����
* @param[in] kpt       ���������
* @param[in] img       ��ȡ�������ͼ��
* @param[in] pattern   Ԥ����õĲ���ģ��
* @param[out] desc     ��������������������õ������ӣ�ά��Ϊ32*8 = 256 bit
*/
void orbDetector::computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc)
{
	//�õ�������ĽǶȣ��û����Ʊ�ʾ������kpt.angle�ǽǶ��ƣ���ΧΪ[0,360)��
	float angle = (float)kpt.angle*factorPI;
	//��������Ƕȵ�����ֵ������ֵ
	float a = (float)cos(angle), b = (float)sin(angle);

	//���ͼ������ָ��
	const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
	//���ͼ���ÿ�е��ֽ���
	const int step = (int)img.step;

	//ԭʼ��BRIEF������û�з��򲻱��ԣ�ͨ������ؼ���ķ��������������ӣ���֮ΪSteer BRIEF�����нϺ���ת��������
	//����أ��ڼ����ʱ����Ҫ������ѡȡ�Ĳ���ģ���е��x�᷽����ת��������ķ���
	//��ò�������ĳ��idx����Ӧ�ĵ�ĻҶ�ֵ,������תǰ����Ϊ(x,y), ��ת������(x',y')�����ǵı任��ϵ:
	// x'= xcos(��) - ysin(��),  y'= xsin(��) + ycos(��)
	// �����ʾ y'* step + x'
	#define GET_VALUE(idx) center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + cvRound(pattern[idx].x*a - pattern[idx].y*b)]        

	//brief��������32*8λ���
	//����ÿһλ���������������ص�Ҷȵ�ֱ�ӱȽϣ�����ÿ�Ƚϳ�8bit�������Ҫ16������㣬��Ҳ����Ϊʲôpattern��Ҫ+=16��ԭ��
	for (int i = 0; i < 32; ++i, pattern += 16)
	{
		int t0, 	//����Ƚϵĵ�1��������ĻҶ�ֵ
			t1,		//����Ƚϵĵ�2��������ĻҶ�ֵ		
			val;	//����������ֽڵıȽϽ����0��1

		t0 = GET_VALUE(0); t1 = GET_VALUE(1);
		val = t0 < t1;							//�����ӱ��ֽڵ�bit0
		t0 = GET_VALUE(2); t1 = GET_VALUE(3);
		val |= (t0 < t1) << 1;					//�����ӱ��ֽڵ�bit1
		t0 = GET_VALUE(4); t1 = GET_VALUE(5);
		val |= (t0 < t1) << 2;					//�����ӱ��ֽڵ�bit2
		t0 = GET_VALUE(6); t1 = GET_VALUE(7);
		val |= (t0 < t1) << 3;					//�����ӱ��ֽڵ�bit3
		t0 = GET_VALUE(8); t1 = GET_VALUE(9);
		val |= (t0 < t1) << 4;					//�����ӱ��ֽڵ�bit4
		t0 = GET_VALUE(10); t1 = GET_VALUE(11);
		val |= (t0 < t1) << 5;					//�����ӱ��ֽڵ�bit5
		t0 = GET_VALUE(12); t1 = GET_VALUE(13);
		val |= (t0 < t1) << 6;					//�����ӱ��ֽڵ�bit6
		t0 = GET_VALUE(14); t1 = GET_VALUE(15);
		val |= (t0 < t1) << 7;					//�����ӱ��ֽڵ�bit7

		//���浱ǰ�Ƚϵĳ����������ӵ�����ֽ�
		desc[i] = (uchar)val;
	}

	//Ϊ�˱���ͳ����е��������ֳ�ͻ�ڣ���ʹ�����֮���ȡ������궨��
	#undef GET_VALUE
}

/**
* @brief ��Ը�����һ��ͼ�񣬼�����ͼ�������
* @param[in] image ������ͼ��
*/
void orbDetector::ComputePyramid(cv::Mat& image)
{
	//��ʼ�������е�ͼ��
	for (int level = 0; level < mnlevels; ++level)
	{
		//��ȡ����ͼ�������ϵ��
		float scale = mvInvScaleFactor[level];
		//���㱾��ͼ������سߴ��С
		cv::Size sz(std::round((float)image.cols*scale), std::round((float)image.rows*scale));

		if (level == 0) mvImagePyramid[level] = image;
		else
		{
			//����һ�������ͼ������趨sz���ŵ���ǰ�㼶
			cv::resize(mvImagePyramid[level - 1],	//����ͼ��
				mvImagePyramid[level], 	            //���ͼ��
				sz, 						        //���ͼ��ĳߴ�
				0, 						            //ˮƽ�����ϵ�����ϵ������0��ʾ�Զ�����
				0,  						        //��ֱ�����ϵ�����ϵ������0��ʾ�Զ�����
				cv::INTER_LINEAR);		            //ͼ�����ŵĲ�ֵ�㷨���ͣ�����������Բ�ֵ�㷨
		}
	}
}

/**
* @brief �԰˲�������������ķ�ʽ������ͼ��������е�������
* @detials ��������vector����˼�ǣ���һ��洢����ĳ��ͼƬ�е����������㣬���ڶ������Ǵ洢ͼ�������������ͼ���vectors of keypoints
* @param[out] allKeypoints ��ȡ�õ�������������
*/
void orbDetector::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints)
{
	//���µ���ͼ�����
	allKeypoints.resize(mnlevels);

	//ͼ��cell�ĳߴ磬�Ǹ������Σ��������Ϊ�߳�in��������
	const float W = 30;

	// ��ÿһ��ͼ��������
	//��������ͼ��
	for (int level = 0; level < mnlevels; ++level)
	{
		const int minBorderX = EDGE_THRESHOLD;
		const int minBorderY = minBorderX;
		const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
		const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

		//�洢��Ҫ����ƽ�������������
		std::vector<cv::KeyPoint> vToDistributeKeys;
		//һ��ض��ǹ����ɼ�����������Ԥ����Ŀռ��С��nfeatures*10
		vToDistributeKeys.reserve(mnfeatures * 10);

		//���������������ȡ��ͼ������ߴ�
		const float width = (maxBorderX - minBorderX);
		const float height = (maxBorderY - minBorderY);

		//���������ڵ�ǰ���ͼ���е�����������
		const int nCols = width / W;
		const int nRows = height / W;
		//����ÿ��ͼ��������ռ����������������
		const int wCell = std::ceil(width / nCols);
		const int hCell = std::ceil(height / nRows);

		//��ʼ����ͼ�����񣬻������п�ʼ������
		for (int i = 0; i < nRows; i++)
		{
			//���㵱ǰ�����ʼ������
			const float iniY = minBorderY + i*hCell;
			//���㵱ǰ�������������꣬�����ǵ��˶����3��Ϊ��cell�߽����ؽ���FAST��������ȡ��
			//ǰ���EDGE_THRESHOLDָ��Ӧ������ȡ������������ڵı߽磬����minBorderY�ǿ����˼���뾶ʱ���ͼ��߽�
			//Ŀ��һ��ͼ������Ĵ�С��25*25
			float maxY = iniY + hCell + 3;

			//�����ʼ����������Ѿ���������Ч��ͼ��߽��ˣ�����ġ���Чͼ����ָԭʼ�ġ�������ȡFAST�������ͼ������
			if (iniY >= maxBorderY - 3) break;

			//���ͼ��Ĵ�С���²��ܹ����û��ֳ��������ͼ��������ô��Ҫί�����һ����
			if (maxY > maxBorderY) maxY = maxBorderY;

			//��ʼ�еı���
			for (int j = 0; j < nCols; j++)
			{
				//�����ʼ��������
				const float iniX = minBorderX + j*wCell;
				//���������������������꣬+6�ĺ����ǰ����ͬ
				float maxX = iniX + wCell + 3;

				//�ж������Ƿ���ͼ����
				//�����ʼ����������Ѿ���������Ч��ͼ��߽��ˣ�����ġ���Чͼ����ָԭʼ�ġ�������ȡFAST�������ͼ������
				//����Ӧ��ͬǰ��������ı߽��Ӧ����Ϊ-3
				if (iniX >= maxBorderX - 3) break;

				//����������Խ����ôί��һ��
				if (maxX > maxBorderX) maxX = maxBorderX;

				// FAST��ȡ��Ȥ��, ����Ӧ��ֵ
				//��������洢���cell�е�������
				std::vector<cv::KeyPoint> vKeysCell;
				//����opencv�Ŀ⺯�������FAST�ǵ�
				cv::FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),	//������ͼ��������ǵ�ǰ��������ͼ���
					vKeysCell,			//�洢�ǵ�λ�õ�����
					miniThFAST,			//�����ֵ
					true);				//ʹ�ܷǼ���ֵ����

				//������ͼ�����ʹ��Ĭ�ϵ�FAST�����ֵû���ܹ���⵽�ǵ�
				if (vKeysCell.empty())
				{
					//��ô��ʹ�ø��͵���ֵ���������¼��
					FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),	//������ͼ��
						vKeysCell,		//�洢�ǵ�λ�õ�����
						minThFAST,		//���͵ļ����ֵ
						true);			//ʹ�ܷǼ���ֵ����
				}

				//��ͼ��cell�м�⵽FAST�ǵ��ʱ��ִ����������
				if (!vKeysCell.empty())
				{
					//�������е�����FAST�ǵ�
					for (std::vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
					{
						//NOTICE ��ĿǰΪֹ����Щ�ǵ�����궼�ǻ���ͼ��cell�ģ���������Ҫ�Ƚ���ָ�����ǰ�ġ�����߽硿�µ�����
						//����������Ϊ������ʹ�ð˲����������������ʱ�򽫻�ʹ�õõ��������
						//�ں��潫�ᱻ����ת����Ϊ�ڵ�ǰͼ�������ͼ������ϵ�µ�����
						(*vit).pt.x += j*wCell;
						(*vit).pt.y += i*hCell;
						//Ȼ������뵽���ȴ������䡰��������������
						vToDistributeKeys.push_back(*vit);
					}//����ͼ��cell�е����е���ȡ������FAST�ǵ㣬���һָ�����������������ǰ��ͼ���µ�����
				}//��ͼ��cell�м�⵽FAST�ǵ��ʱ��ִ����������
			}//��ʼ����ͼ��cell����
		}//��ʼ����ͼ��cell����

		//����һ���Ե�ǰͼ��������������������
		std::vector<cv::KeyPoint> & keypoints = allKeypoints[level];
		//���ҵ������СΪ����ȡ�������������������Ȼ����Ҳ�������˵ģ���Ϊ���������е������㶼������һ��ͼ������ȡ�����ģ�
		keypoints.reserve(mnfeatures);

		// ����mnFeatuvector<KeyPoint> & keypoints = allKeypoints[level];resPerLevel,���ò����Ȥ����,������������޳�
		//����ֵ��һ���������������vector�����������޳���ı���������������
		//�õ�������������꣬�������ڵ�ǰͼ����������
		keypoints = DistributeOctTree(vToDistributeKeys, 			//��ǰͼ����ȡ�����������㣬Ҳ���ǵȴ��޳���������																
			minBorderX, maxBorderX,		                            //��ǰͼ��ͼ��ı߽�
			minBorderY, maxBorderY,
			mv_nFeaturesPerLevel[level], 	                        //ϣ�����������ĵ�ǰ��ͼ������������
			level);						                            //��ǰ��ͼ�����ڵ�ͼ��

		//PATCH_SIZE�Ƕ��ڵײ�ĳ�ʼͼ����˵�ģ�����Ҫ���ݵ�ǰͼ��ĳ߶����ű����������ŵõ����ź��PATCH��С ��������ķ�������й�
		const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

		//��ȡ�޳����̺�����������������Ŀ
		const int nkps = keypoints.size();
		//Ȼ��ʼ������Щ�����㣬�ָ����ڵ�ǰͼ��ͼ������ϵ�µ�����
		for (int i = 0; i < nkps; i++)
		{
			//��ÿһ�����������������㣬�ָ�������ڵ�ǰͼ�㡰��Ե����ͼ���¡�������ϵ������
			keypoints[i].pt.x += minBorderX;
			keypoints[i].pt.y += minBorderY;
			//��¼��������Դ��ͼ�������ͼ��
			keypoints[i].octave = level;
			//��¼���㷽���patch�����ź��Ӧ�Ĵ�С�� �ֱ�����Ϊ������뾶
			keypoints[i].size = scaledPatchSize;
		}
	}

	//Ȼ�������Щ������ķ�����Ϣ��ע�����ﻹ�Ƿֲ�����
	for (int level = 0; level < mnlevels; ++level)
	{
		computeOrientation(mvImagePyramid[level],	//��Ӧ��ͼ���ͼ��
			allKeypoints[level], 	                //���ͼ������ȡ����������������������
			mv_umax);					            //�Լ�PATCH�ĺ�����߽�
	}
}

/**
* @brief ����������ķ���
* @param[in] image                 ���������ڵ�ǰ��������ͼ��
* @param[in & out] keypoints       ����������
* @param[in] umax                  ÿ������������ͼ�������ÿ�еı߽� u_max ��ɵ�vector
*/
void orbDetector::computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax)
{
	// �������е�������
	for (auto& keypoint: keypoints)
	{
		// ����IC_Angle �����������������ķ���
		keypoint.angle = IC_Angle(image, 			//���������ڵ�ͼ���ͼ��
			keypoint.pt, 	                        //������������ͼ���е�����
			umax);			                        //ÿ������������ͼ�������ÿ�еı߽� u_max ��ɵ�vector
	}
}

/**
* @brief ����������ڼ���������ķ��������Ƿ��ؽǶ���Ϊ����
* ���������㷽����Ϊ��ʹ����ȡ�������������ת�����ԡ�
* �����ǻҶ����ķ����Լ������ĺͻҶ����ĵ�������Ϊ�������㷽��
* @param[in] image     Ҫ���в�����ĳ�������ͼ��
* @param[in] pt        ��ǰ�����������
* @param[in] u_max     ͼ����ÿһ�е�����߽� u_max
* @return float        ����������ĽǶȣ���ΧΪ[0,360)�Ƕȣ�����Ϊ0.3��
*/
float orbDetector::IC_Angle(const cv::Mat& image, cv::Point2f pt, const std::vector<int> & u_max)
{
	//ͼ��ľأ�ǰ���ǰ���ͼ����y�����Ȩ�������ǰ���ͼ����x�����Ȩ
	int m_01 = 0, m_10 = 0;

	//���������������ڵ�ͼ�������ĵ�����Ҷ�ֵ��ָ��center
	const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

	//����v=0�����ߵļ�����Ҫ����Դ�
	//��������������Ϊ�Գ��ᣬ�ɶԱ�������������PATCH_SIZE����������
	for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
		//ע�������center�±�u�����Ǹ��ģ�����ˮƽ���ϵ����ذ�x���꣨Ҳ����u���꣩��Ȩ
		m_10 += u * center[u];

	// Go line by line in the circular patch  
	//�����step1��ʾ���ͼ��һ�а������ֽ��������ο�[https://blog.csdn.net/qianqing13579/article/details/45318279]
	int step = (int)image.step1();
	//ע����������v=0������Ϊ�Գ��ᣬȻ��ԳƵ�ÿ�ɶԵ�����֮����б�������������ӿ��˼����ٶ�
	for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
	{
		//����m_01Ӧ����һ��һ�еؼ���ģ��������ڶԳ��Լ�����x,y������ԭ�򣬿���һ�μ�������
		int v_sum = 0;
		// ��ȡĳ�����غ���������Χ��ע�������ͼ�����Բ�εģ�
		int d = u_max[v];
		//�����귶Χ�ڰ������ر�����ʵ����һ�α���2��
		// ����ÿ�δ�������������꣬�������·�Ϊ(x,y),�������Ϸ�Ϊ(x,-y) 
		// ����ĳ�δ�����������㣺m_10 = �� x*I(x,y) =  x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
		// ����ĳ�δ�����������㣺m_01 = �� y*I(x,y) =  y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
		for (int u = -d; u <= d; ++u)
		{
			//�õ���Ҫ���м�����ͼ���������ػҶ�ֵ
			//val_plus�����������·�x=uʱ�ĵ����ػҶ�ֵ
			//val_minus�����������Ϸ�x=uʱ�����ػҶ�ֵ
			int val_plus = center[u + v*step], val_minus = center[u - v*step];
			//��v��y�ᣩ�ϣ�2���������ػҶ�ֵ֮��
			v_sum += (val_plus - val_minus);
			//u�ᣨҲ����x�ᣩ��������u�����Ȩ�ͣ�u����Ҳ���������ţ����൱��ͬʱ��������
			m_10 += u * (val_plus + val_minus);
		}
		//����һ���ϵĺͰ���y�����Ȩ
		m_01 += v * v_sum;
	}

	//Ϊ�˼ӿ��ٶȻ�ʹ����fastAtan2()���������Ϊ[0,360)�Ƕȣ�����Ϊ0.3��
	return cv::fastAtan2((float)m_01, (float)m_10);
}

/**
* @brief ����ȡ���ڵ�ֳ�4���ӽڵ㣬ͬʱҲ���ͼ������Ļ��֡�����������Ļ��֣��Լ���ر�־λ����λ
*
* @param[in & out] n1  ��ȡ���ڵ�1������
* @param[in & out] n2  ��ȡ���ڵ�1������
* @param[in & out] n3  ��ȡ���ڵ�1������
* @param[in & out] n4  ��ȡ���ڵ�1������
*/
void DetectorNode::DivideNode(DetectorNode &n1,
	DetectorNode &n2,
	DetectorNode &n3,
	DetectorNode &n4)
{
	//�õ���ǰ��ȡ���ڵ�����ͼ�������һ�볤����Ȼ�����Ҫȡ��
	const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
	const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

	//����Ĳ�����ͬС�죬��һ��ͼ��������ϸ�ֳ�Ϊ�ĸ�Сͼ������
	//n1 �洢��������ı߽�
	n1.UL = UL;
	n1.UR = cv::Point2i(UL.x + halfX, UL.y);
	n1.BL = cv::Point2i(UL.x, UL.y + halfY);
	n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
	//�����洢�ڸýڵ��Ӧ��ͼ����������ȡ�������������vector
	n1.vKeys.reserve(vKeys.size());

	//n2 �洢��������ı߽�
	n2.UL = n1.UR;
	n2.UR = UR;
	n2.BL = n1.BR;
	n2.BR = cv::Point2i(UR.x, UL.y + halfY);
	n2.vKeys.reserve(vKeys.size());

	//n3 �洢��������ı߽�
	n3.UL = n1.BL;
	n3.UR = n1.BR;
	n3.BL = BL;
	n3.BR = cv::Point2i(n1.BR.x, BL.y);
	n3.vKeys.reserve(vKeys.size());

	//n4 �洢��������ı߽�
	n4.UL = n3.UR;
	n4.UR = n2.BR;
	n4.BL = n3.BR;
	n4.BR = BR;
	n4.vKeys.reserve(vKeys.size());

	//Associate points to childs
	//������ǰ��ȡ���ڵ��vkeys�д洢��������
	for (size_t i = 0; i<vKeys.size(); i++)
	{
		//��ȡ������������
		const cv::KeyPoint &kp = vKeys[i];
		//�ж�����������ڵ�ǰ��������ȡ���ڵ�ͼ����ĸ����򣬸��ϸ��˵�������Ǹ���ͼ������
		//Ȼ��ͽ����������׷�ӵ��Ǹ���������ȡ���ڵ��vkeys��
		if (kp.pt.x<n1.UR.x)
		{
			if (kp.pt.y<n1.BR.y)
				n1.vKeys.push_back(kp);
			else
				n3.vKeys.push_back(kp);
		}
		else if (kp.pt.y<n1.BR.y)
			n2.vKeys.push_back(kp);
		else
			n4.vKeys.push_back(kp);
	}//������ǰ��ȡ���ڵ��vkeys�д洢��������

	 //�ж�ÿ������������ȡ���ڵ����ڵ�ͼ�������������Ŀ�����Ƿ�����ӽڵ����������Ŀ����Ȼ�������
	 //�����ж��Ƿ���Ŀ����1��Ŀ����ȷ������ڵ㻹�ܲ��������½��з���
	if (n1.vKeys.size() == 1)
		n1.bNoMore = true;
	if (n2.vKeys.size() == 1)
		n2.bNoMore = true;
	if (n3.vKeys.size() == 1)
		n3.bNoMore = true;
	if (n4.vKeys.size() == 1)
		n4.bNoMore = true;
}

/**
* @brief ʹ���Ĳ�������һ��ͼ�������ͼ���е����������ƽ���ͷַ�
*
* @param[in] vToDistributeKeys     �ȴ����з��䵽�Ĳ����е�������
* @param[in] minX                  ��ǰͼ���ͼ��ı߽磬���궼���ڡ��뾶����ͼ������ϵ�µ�����
* @param[in] maxX
* @param[in] minY
* @param[in] maxY
* @param[in] N                     ϣ����ȡ�������������
* @param[in] level                 ָ���Ľ�����ͼ�㣬��δʹ��
* @return vector<cv::KeyPoint>     �Ѿ����ȷ�ɢ�õ�������vector����
*/
std::vector<cv::KeyPoint> orbDetector::DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
	const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
	// Step 1 ���ݿ�߱�ȷ����ʼ�ڵ���Ŀ
	//����Ӧ�����ɵĳ�ʼ�ڵ���������ڵ������nIni�Ǹ��ݱ߽�Ŀ�߱�ֵȷ���ģ�һ����1����2
	int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));
	if (nIni == 0) nIni = 1;

	//һ����ʼ�Ľڵ��x�����ж��ٸ�����
	const float hX = static_cast<float>(maxX - minX) / nIni;

	//�洢����ȡ���ڵ������
	std::list<DetectorNode> lNodes;

	//�洢��ʼ��ȡ���ڵ�ָ���vector
	std::vector<DetectorNode*> vpIniNodes;

	//�����������С
	vpIniNodes.resize(nIni);

	// Step 2 ���ɳ�ʼ��ȡ���ڵ�
	for (int i = 0; i < nIni; i++)
	{
		//����һ����ȡ���ڵ�
		DetectorNode ni;

		//������ȡ���ڵ��ͼ��߽�
		//ע���������ȡFAST�ǵ�������ͬ�����ǡ��뾶����ͼ�񡱣������������0 ��ʼ 
		ni.UL = cv::Point2i(hX*static_cast<float>(i), 0);       //UpLeft
		ni.UR = cv::Point2i(hX*static_cast<float>(i + 1), 0);   //UpRight
		ni.BL = cv::Point2i(ni.UL.x, maxY - minY);		        //BottomLeft
		ni.BR = cv::Point2i(ni.UR.x, maxY - minY);              //BottomRight
															   
		ni.vKeys.reserve(vToDistributeKeys.size());             //����vkeys��С

		//���ղ����ɵ���ȡ�ڵ���ӵ�������
		//��Ȼ�����ni�Ǿֲ��������������������push_back()�ǿ������������ݵ�һ���µĶ�����Ȼ������ӵ��б���
		//���Ե��������˳�֮��������ڴ治���Ϊ��Ұָ�롱
		lNodes.push_back(ni);
		//�洢�����ʼ����ȡ���ڵ���
		vpIniNodes[i] = &lNodes.back();
	}

	// Step 3 ����������䵽����ȡ���ڵ���
	for (size_t i = 0; i < vToDistributeKeys.size(); i++)
	{
		//��ȡ������������
		const cv::KeyPoint &kp = vToDistributeKeys[i];
		//��������ĺ���λ�ã�����������Ǹ�ͼ���������ȡ���ڵ㣨�������ȡ���ڵ㣩
		vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
	}

	for (auto& lit: lNodes)
	{
		//�����ʼ����ȡ���ڵ������䵽�����������Ϊ1, ��ô�ͱ�־λ��λ����ʾ�˽ڵ㲻���ٷ�
		if (lit.vKeys.size() <=1 ) lit.bNoMore = true;
	}

	//������־λ���
	bool bFinish = false;

	//����һ��vector���ڴ洢�ڵ��vSize�;����
	//���������¼����һ�η���ѭ���У���Щ�����ټ������з��ѵĽڵ��а�������������Ŀ������
	std::vector<std::pair<int, DetectorNode*> > vSizeAndPointerToNode;

	//������С���������˼��һ����ʼ���ڵ㽫�����ѡ���Ϊ�ĸ�
	vSizeAndPointerToNode.reserve(lNodes.size() * 4);

	//�洢��ȡ���ڵ���б���ʵ����˫��������һ��������,���Բο�[http://www.runoob.com/cplusplus/cpp-overloading.html]
	//����������ṩ�˷����ܽڵ��б�ķ�ʽ����Ҫ���cpp�ļ����з���
	std::list<DetectorNode>::iterator lit;

	// Step 5 �����Ĳ���������ͼ����л������򣬾��ȷ���������
	while (!bFinish)
	{
		//���浱ǰ�ڵ������prev���������Ϊ���������ȽϺ�
		int prevSize = lNodes.size();

		//���¶�λ������ָ���б�ͷ��
		lit = lNodes.begin();

		//��Ҫչ���Ľڵ���������һֱ�����ۼƣ�������
		int nToExpand = 0;

		//��Ϊ����ѭ���У�ǰ���ѭ�����п�����Ⱦ������������������
		//�������Ҳֻ��ͳ����ĳһ��ѭ���еĵ�
		//���������¼����һ�η���ѭ���У���Щ�����ټ������з��ѵĽڵ��а�������������Ŀ������
		vSizeAndPointerToNode.clear();

		//��Ŀǰ����������л���
		//��ʼ�����б������е���ȡ���ڵ㣬�����зֽ���߱���
		while (lit != lNodes.end())
		{
			//�����ȡ���ڵ�ֻ��һ�������㣬
			if (lit->bNoMore)
			{
				//��ô��û�б�Ҫ�ٽ���ϸ����
				lit++;
				//������ǰ�ڵ㣬������һ��
				continue;
			}
			else
			{
				//�����ǰ����ȡ���ڵ���г���һ���������㣬��ô��Ҫ���м�������
				DetectorNode n1, n2, n3, n4;

				//��ϸ�ֳ��ĸ�������
				lit->DivideNode(n1, n2, n3, n4);

				//�������ֳ��������������������㣬��ô�ͽ����������Ľڵ���ӵ���ȡ���ڵ���б���
				//ע������������ǣ��������㼴��
				if (n1.vKeys.size() > 0)
				{
					//ע������Ҳ����ӵ��б�ǰ���
					lNodes.push_front(n1);

					//���ж���������ȡ���ڵ��е���������Ŀ�Ƿ����1
					if (n1.vKeys.size() > 1)
					{
						//����г���һ���������㣬��ô��չ���Ľڵ������1
						nToExpand++;

						//���������������Ŀ�ͽڵ�ָ�����Ϣ
						vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));

						//lNodes.front().iter ��ǰ��ĵ�����lit ��ͬ
						//lNodes.front().iter��node�ṹ�����һ��ָ��������¼�ڵ��λ��
						//���ں���ɾ���ýڵ�ʹ��
						lNodes.front().iter = lNodes.begin();
					}
				}
				//����Ĳ���������ͬ�ģ����ﲻ��׸��
				if (n2.vKeys.size() > 0)
				{
					lNodes.push_front(n2);
					if (n2.vKeys.size() > 1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
						lNodes.front().iter = lNodes.begin();
					}
				}
				if (n3.vKeys.size() > 0)
				{
					lNodes.push_front(n3);
					if (n3.vKeys.size() > 1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
						lNodes.front().iter = lNodes.begin();
					}
				}
				if (n4.vKeys.size() > 0)
				{
					lNodes.push_front(n4);
					if (n4.vKeys.size() > 1)
					{
						nToExpand++;
						vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
						lNodes.front().iter = lNodes.begin();
					}
				}
				//�����ĸ�ڵ�expand֮��ʹ��б���ɾ�����ˣ��ܹ����з��Ѳ���˵��������һ���ӽڵ���������������������>1��
				// ���ѷ�ʽ�Ǻ�ӵĽڵ��ȷ��ѣ��ȼӵĺ����
				lit = lNodes.erase(lit);
			}//�жϵ�ǰ�������Ľڵ����Ƿ��г���һ����������
		}//�����б��е�������ȡ���ڵ�

		//ֹͣ������̵���������������������һ�����ɣ�
		//1����ǰ�Ľڵ����Ѿ�������Ҫ�����������
		//2����ǰ���еĽڵ��ж�ֻ����һ��������
		if ((int)lNodes.size() >= N 				//�ж��Ƿ񳬹���Ҫ�����������
			|| (int)lNodes.size() == prevSize)	    //prevSize�б�����Ƿ���֮ǰ�Ľڵ�������������֮ǰ�ͷ���֮����ܽڵ����һ����˵����ǰ���е�
												    //�ڵ�������ֻ��һ�������㣬�Ѿ����ܹ���ϸ����
		{
			//ֹͣ��־��λ
			bFinish = true;
		}
		
		//����չ�����ӽڵ����nToExpand x3������Ϊһ����֮�󣬻�ɾ��ԭ�������ڵ㣬���Գ���3
		/**
		* ע�⵽�������nToExpand������ǰ���ִ�й�������һֱ�����ۼ�״̬�ģ������Ϊ���������̫�٣������������else-if���ֽ�����һ������ı���
		* list�Ĳ���֮��lNodes.size()�����ˣ�����nToExpandҲ�����ˣ��������ںܶ�β���֮������ı��ʽ��
		* ((int)lNodes.size()+nToExpand*3)>N
		* ��ܿ�ͱ����㣬���Ǵ�ʱֻ����һ�ζ�vSizeAndPointerToNode�е���з��ѵĲ����ǿ϶������ģ�
		* �����У����������for������ֻҪִ��һ�ξ������㣬�������������ǵġ������������Ӧ���Ƿ��Ѻ���ֵĽڵ������������û�������㣬��˽�for
		* ѭ��������һ��whileѭ�����棬ͨ���ٴν���forѭ�����ٷ���һ�ν��������⡣
		* */
		if ((lNodes.size() + nToExpand*3) > N)
		{
			//����ٷ���һ����ô��Ŀ��Ҫ���ˣ�������취������ʹ��ոմﵽ���߳���Ҫ������������ʱ���˳�
			//�����nToExpand��vSizeAndPointerToNode����һ��ѭ����һ��ѭ���Ĺ�ϵ������ǰ�����ۼƼ���������ֻ����ĳһ��ѭ����
			//һֱѭ����ֱ��������־λ����λ
			while (!bFinish)
			{
				//��ȡ��ǰ��list�еĽڵ����
				prevSize = lNodes.size();

				//������Щ�����Է��ѵĽڵ����Ϣ, ���������
				std::vector<std::pair<int, DetectorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
				//���
				vSizeAndPointerToNode.clear();

				// ����Ҫ���ֵĽڵ�������򣬶�pair�Եĵ�һ��Ԫ�ؽ�������Ĭ���Ǵ�С��������
				// ���ȷ����������Ľڵ㣬ʹ���������ܼ������������ٵ�������
				//! ע��������������ǳ���Ҫ���ᵼ��ÿ���������������㶼��һ��������ʹ�� stable_sort
				sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

				//��������洢��pair�Ե�vector��ע���ǴӺ���ǰ����
				for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
				{
					DetectorNode n1, n2, n3, n4;
					//��ÿ����Ҫ���з��ѵĽڵ���з���
					vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

					// Add childs if they contain points
					//��ʵ����Ľڵ����˵�Ƕ����ӽڵ��ˣ�ִ�к�ǰ��һ���Ĳ���
					if (n1.vKeys.size() > 0)
					{
						lNodes.push_front(n1);
						if (n1.vKeys.size() > 1)
						{
							//��Ϊ���ﻹ�ж���vSizeAndPointerToNode�Ĳ���������ǰ��Żᱸ��vSizeAndPointerToNode�е�����
							//Ϊ���ܵġ���������һ��forѭ����׼��
							vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
							lNodes.front().iter = lNodes.begin();
						}
					}
					if (n2.vKeys.size() > 0)
					{
						lNodes.push_front(n2);
						if (n2.vKeys.size() > 1)
						{
							vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
							lNodes.front().iter = lNodes.begin();
						}
					}
					if (n3.vKeys.size() > 0)
					{
						lNodes.push_front(n3);
						if (n3.vKeys.size() > 1)
						{
							vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
							lNodes.front().iter = lNodes.begin();
						}
					}
					if (n4.vKeys.size() > 0)
					{
						lNodes.push_front(n4);
						if (n4.vKeys.size() > 1)
						{
							vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
							lNodes.front().iter = lNodes.begin();
						}
					}

					//ɾ��ĸ�ڵ㣬��������ʵӦ����һ���ӽڵ�
					lNodes.erase(vPrevSizeAndPointerToNode[j].second->iter);

					//�ж����Ƿ񳬹�����Ҫ�������������ǵĻ����˳������ǵĻ��ͼ���������ѹ��̣�ֱ���ոմﵽ���߳���Ҫ������������
					//���ߵ�˼����ʵ���������ģ��ٷ�����һ��֮���ж���һ�η����Ƿ�ᳬ��N�����������ô�ͷ��Ĵ󵨵�ȫ�����з��ѣ���Ϊ����һ���ж����
					//�������ٶȻ���΢��һЩ�����������ô������������������һ�η���
					if ((int)lNodes.size() >= N)
						break;
				}//����vPrevSizeAndPointerToNode��������ָ����node���з��ѣ�ֱ���ոմﵽ���߳���Ҫ������������
				
				//����������Ӧ����һ��forѭ�����ܹ���ɽ��������ˣ�����������Ŀ����ǣ���Щ�ӽڵ����ڵ������û�������㣬��˺��п���һ��forѭ��֮��
				//����Ŀ���ǲ��ܹ�����Ҫ�����Ի�����Ҫ�жϽ���������������һ��
				//�ж��Ƿ�ﵽ��ֹͣ����
				if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
					bFinish = true;
			}//һֱ����nToExpand�ۼӵĽڵ���ѹ��̣�ֱ�����Ѻ��nodes��Ŀ�ոմﵽ���߳���Ҫ�����������Ŀ
		}//�����η��Ѻ�ﲻ���������������ٽ���һ�������ķ���֮��Ϳ��Դﵽ��������ʱ
	}// ������Ȥ��ֲ�,����4����������ͼ����л�������

	// Step 7 ����ÿ��������Ӧֵ����һ����Ȥ��
	//ʹ�����vector���洢���Ǹ���Ȥ��������Ĺ��˽��
	std::vector<cv::KeyPoint> vResultKeys;

	//����������СΪҪ��ȡ����������Ŀ
	vResultKeys.reserve(mnfeatures);

	//��������ڵ�����
	for (std::list<DetectorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
	{
		//�õ�����ڵ������е��������������
		std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;

		//�õ�ָ���һ���������ָ�룬������Ϊ�����Ӧֵ��Ӧ�Ĺؼ���
		cv::KeyPoint* pKP = &vNodeKeys[0];

		//�õ�1���ؼ�����Ӧֵ��ʼ�������Ӧֵ
		float maxResponse = pKP->response;

		//��ʼ��������ڵ������е������������е������㣬ע���Ǵ�1��ʼӴ��0�Ѿ��ù���
		for (size_t k = 1; k < vNodeKeys.size(); k++)
		{
			//���������Ӧֵ
			if (vNodeKeys[k].response > maxResponse)
			{
				//����pKPָ����������Ӧֵ��keypoints
				pKP = &vNodeKeys[k];
				maxResponse = vNodeKeys[k].response;
			}
		}

		//������ڵ������е���Ӧֵ����������������ս������
		vResultKeys.push_back(*pKP);
	}

	//�������ս�����������б����з��ѳ����������У����������Ȥ����Ӧֵ����������
	return vResultKeys;
}

}