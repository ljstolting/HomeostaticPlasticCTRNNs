//--------------------------------------------------
// ALIFE 2023 Timescale Separation Sweep Experiment
//--------------------------------------------------

#include "CTRNN.h"

// Global constants
const double TransientDuration = 500;  //in seconds, transient passed before checking frequency of osc
const double RunDuration = 50;         //time in seconds to test 
const double StepSize = 0.01;
const double maxNetworkSize = 20;
const double minNetworkSize = 1;
// const double minbdry = 0; //when searching over a variety of HP meta parameter values
// const double maxbdry = .5;
const double WR = 16.0;          //Weight range (+/-)
const double BR = 16.0;          //Bias Range (+/-)
const double TMIN = 0.5;
const double TMAX = 10.0;
const double Circuits = 100000;     //How many circuits in sample
const double Repetitions = 3;    //How many initial state conditions will you start from for each circuit

const double MinSearchValue = -1.0;
const double MaxSearchValue = 1.0;

double clip(double x, double min, double max)
{
	double temp;
	temp = ((x > min)?x:min);
	return (temp < max)?temp:max;
}

double MapSearchParameter(double x, double min, double max, double clipmin = -1.0e99, double clipmax = 1.0e99)
{
	double m = (max - min)/(MaxSearchValue - MinSearchValue);
	double b = min - m * MinSearchValue;
	return clip(m * x + b,clipmin,clipmax);
}

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(int n, TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= n; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= n; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
}

// The main program
int main(int argc, char* argv[])
{
    std::string const & FileName1 = argv[1];
    ofstream OGdimension(FileName1);
    std::string const & FileName2 = argv[2];
    ofstream trackneuralpars(FileName2);
    std::string const & FileName3 = argv[3];
    ofstream HPondimension(FileName3);
    std::string const & FileName4 = argv[4];
    ofstream HPoffdimension(FileName4);
    std::string const & FileName5 = argv[5];
    ofstream neuralstates(FileName5);
    std::string const & FileName6 = argv[6];
    ofstream centercrossingdistance(FileName6);

    // Set random number generator and seed
    RandomState rs;
    long seed=-time(0);
    rs.SetRandomSeed(seed);

		// Plasticity and CTRNN size parameters (comment out whichever one you are varying over the x-axis)
		int WS = 1; 				// Window Size of Plastic Rule (in steps size) (so 1 is no window)
		double B = 0.25; 		// Plasticity Low Boundary
    //int n = 2;        //circuit size
		double BT = 20.0;		// Bias Time Constant
		double WT = 40.0;		// Weight Time Constant

    // For each boundray size
    for (int n = minNetworkSize; n <= maxNetworkSize; n ++) {
      cout<< "Size " << n << endl;
      // Number of parameters
      int VectSize = n*n + 2*n;

      // Number of circuits that "oscillate"
      int counterOG = 0;

      // Number of circuits that "oscillate" while HP turned on
      int counterHPon = 0;

      // Number of circuits that "oscillate" after HP is turned back off
      int counterHPoff = 0;
    }
}

