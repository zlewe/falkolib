#include "python_wrapper.h"

#include <pybind11/stl.h>

namespace py = pybind11;

class LaserMatcher{
    public:
        LaserScan scan1;
        LaserScan scan2;


        FALKOExtractor fe;

        vector<FALKO> keypoints1;
	    vector<FALKO> keypoints2;

        vector<BSC> bscDesc1;
	    vector<BSC> bscDesc2;

        LaserMatcher(){
            scan1 = LaserScan(0, 2.0 * M_PI, 360/0.2);
            scan2 = LaserScan(0, 2.0 * M_PI, 360/0.2);
    
            fe = FALKOExtractor();
            fe.setMinExtractionRange(0);
            fe.setMaxExtractionRange(30);
            fe.enableSubbeam(true);
            fe.setNMSRadius(0.1);
            fe.setNeighB(0.07);
            fe.setBRatio(2.5);
            fe.setGridSectors(16);;
        }

        void insertScan1(const vector<double>& ranges){
            scan1.fromRanges(ranges);
        }

        void insertScan2(const vector<double>& ranges){
            scan2.fromRanges(ranges);
        }

        vector<int> computeDesc(){
            fe.extract(scan1, keypoints1);
	        fe.extract(scan2, keypoints2);

            vector<int> ret;
            ret.push_back(keypoints1.size());
            ret.push_back(keypoints2.size());

            BSCExtractor<FALKO> bsc(16,8);
            
            bsc.compute(scan1, keypoints1, bscDesc1);
	        bsc.compute(scan2, keypoints2, bscDesc2);

            return ret;
        }

        std::vector<std::pair<int, int>> NMscanMatcher(){
            NNMatcher<FALKO> matcher;
            matcher.setDistanceThreshold(0.1);
            std::vector<std::pair<int, int> > assoNN;
            // std::cout << "num matching NN: " << matcher.match(keypoints1, keypoints2, assoNN) << endl;
            for (auto& match : assoNN) {
                if (match.second >= 0) {
                    int i1 = match.first;
                    int i2 = match.second;
                    // std::cout << "i1: " << i1 << "\ti2: " << i2 << "\t keypoints distance: " << (keypoints1[i1].distance(keypoints2[i2])) << endl;
                }
            }
            return assoNN;
        } 
};

namespace falkopy {

py::array_t<double_t, py::array::c_style> getTransform(LaserMatcher& lm) {
	
	Eigen::Affine2d transformNN;
    std::vector<std::pair<int, int>> assoNN = lm.NMscanMatcher();
	computeTransform(lm.keypoints1, lm.keypoints2, assoNN, transformNN);

    return py::cast(transformNN.inverse().matrix());
}

py::array_t<int64_t, py::array::c_style> compute_desc(LaserMatcher& lm) {

    return py::cast(lm.computeDesc());
}

}

PYBIND11_MODULE(falkopy, m) {
  using namespace falkopy;
  m.doc() = "falkolib bindings";

  py::class_<LaserMatcher>(m, "LaserMatcher")
      .def(py::init<>())
      .def("insert_scan1", &LaserMatcher::insertScan1)
      .def("insert_scan2", &LaserMatcher::insertScan2)
      .def("compute_desc", compute_desc)
      .def("get_transform", getTransform);
}