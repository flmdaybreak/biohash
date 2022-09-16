#include <iostream>    
#include <atomic>
#include <random>
#include <bitset>
#include <Eigen/Dense>    

#define DIMENSION 512

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using namespace Eigen;
using namespace std;

typedef Matrix<double,1,Dynamic> MatrixType;
typedef Map<MatrixType> MapType;
typedef Matrix<float, Dynamic, Dynamic> RowMatrixXf;

float generate_random(float dummy){
        static std::default_random_engine e(111111111);
        static std::normal_distribution<float> n(0,1);
        return n(e);
}

MatrixXf init_Q(){
        static std::atomic<bool> inited = false;
        static MatrixXf Q;// Q =  qr.householderQ();
        if(inited == false) {
            // static MatrixXf Q;
            MatrixXf mat_rand = MatrixXf::Zero(DIMENSION,DIMENSION).unaryExpr(std::ptr_fun(generate_random));
            HouseholderQR<MatrixXf> qr;
            qr.compute(mat_rand);
            MatrixXf R = qr.matrixQR().triangularView<Upper>();
            Q =  qr.householderQ();
            inited = true;
        }

        return Q;
}

 std::bitset<DIMENSION> get_biohash(std::vector<float>& input, MatrixXf &Q){
        Map<RowMatrixXf> eig(input.data(), 1, input.size());
        std::bitset<DIMENSION>bit_biohash_res;
        RowMatrixXf res = eig * Q;
        int i = 0;
        std::vector<float> biohashs(res.data(), res.data() + res.rows() * res.cols());
        float mean_value  = std::accumulate(std::begin(biohashs), std::end(biohashs), 0.0)/input.size();
        for(auto tmp : biohashs){
            if(tmp > mean_value){
                bit_biohash_res[i] = 1;
            }else{
                bit_biohash_res[i] = 0;
            }
            i++;
        }
        return bit_biohash_res;
}

 std::bitset<DIMENSION> cal_biohash(vector<float>& data){
	 MatrixXf Q = init_Q();
         return get_biohash(data,Q);
 
}

int hamming_distance(std::bitset<DIMENSION>& one,std::bitset<DIMENSION>& two){
	std::bitset<DIMENSION> res = one^two;
	return res.count();

}

int  main(int argc, char* argv[]){
	return 0;
}
