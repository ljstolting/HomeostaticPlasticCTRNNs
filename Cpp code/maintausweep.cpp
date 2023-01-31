//--------------------------------------------------
// ALIFE 2023 Timescale Separation Sweep Experiment
//
// How does HP's participation in oscillations change
// as a funciton of the degree of separation between
// its timescale and that of the neural states?
//--------------------------------------------------

#include "CTRNN.h"

// Global constants
const double TransientDuration = 500;  //in seconds, transient passed before checking frequency of osc
const double RunDuration = 50;         //time in seconds to test osc frequency
const double StepSize = 0.01;
const double WR = 16.0;          //Weight range (+/-)
const double BR = 16.0;          //Bias Range (+/-)
const double TMIN = 1;      //Neural taus are held constant 
const double TMAX = 1;
const double Circuits = 250;     //How many HPCTRNNs oscillating at the target frequncy do we want to start off with (honestly we should probably evolve them instead of random generation)
//const double Repetitions = 3;    //How many initial state conditions will you start from for each circuit (important for varying where in the cycle HP turns off)

const double MinSearchValue = -1.0;  //for random CTRNN generation
const double MaxSearchValue = 1.0;

const double HPtaumin = .5; //Minimum value of tauw and taub
const double HPtaumax = 50; //Max value of tauw and taub
const double HPtausample = 1; //Value of HP tau that we want to use to generate the sample of HPCTRNNs in the range 
const double TargetFrequency = 0.855556/2;  //Target frequency obtained by halving the highest evolved freq for a 3-neuron-circuit
const double TargetMargin = .05; //How much are "good" circuits allowed to vary from the threshold to be included in the initial sample?

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

// Plasticity and CTRNN size parameters
int WS = 1; 				// Window Size of Plastic Rule (in steps size) (so 1 is no window)
double B = 0.25; 		// Plasticity Low Boundary
int n = 3;          //circuit size
int VectSize = n*n + 2*n; //Number of circuit parameters

// Run Circuit and Determine Frequency of oscillation (throw zero if not oscillating) -- obtain from Eduardo's code
void frequencytest(TVector<double> &phen, double HPtau, TVector<double> &freq)
{// Generate circuit and Set phenotype to circuit
CTRNN c(n,WS,B,HPtau,HPtau,WR,BR);

int k = 1;
// Time-constants
for (int i = 1; i <= n; i++) {
  c.SetNeuronTimeConstant(i, phen(k));
  k++;
}
// Bias
for (int i = 1; i <= n; i++) {
  c.SetNeuronBias(i, phen(k));
  k++;
}
// Weights
for (int i = 1; i <= n; i++) {
  for (int j = 1; j <= n; j++) {
    c.SetConnectionWeight(i, j, phen(k));
    k++;
  }
}

/// Randomize Outputs
c.RandomizeCircuitOutput(0.1, 0.9);

//pass transient
for (double time = StepSize; time <= TransientDuration; time += StepSize) {
  c.EulerStep(StepSize);
}

TVector<double> pastNeuronOutput(1,n);
for (int i=1;i<=n;i++){
pastNeuronOutput[i] = c.NeuronOutput(i);
}
TVector<double> marker(1,n);
marker = pastNeuronOutput;

TVector<double> numbercycles(1,n);
numbercycles.FillContents(0);

//test for oscillation frequency of each neuron (count the number of times each neuron passes from above to below their starting value)
for (double time = StepSize; time<=RunDuration; time += StepSize){
  c.EulerStep(StepSize);
  for (int i=1;i<=n;i++){
    if (c.NeuronOutput(i)<marker[i] && pastNeuronOutput[i]>marker[i]){
      numbercycles[i] ++;
    }
    pastNeuronOutput[i] = c.NeuronOutput(i);
  }
}
// Divide by test duration in seconds
for (int i=1;i<=n;i++){
  freq[i] = numbercycles[i]/RunDuration;
}

}

// The main program
int main(int argc, char* argv[])
{
    std::string const & FileName1 = argv[1];
    ofstream basegenomes(FileName1);
    std::string const & FileName2 = argv[2];
    ofstream HPonfreq(FileName2);
    // std::string const & FileName3 = argv[3];
    // ofstream neuralstates(FileName3);
    // std::string const & FileName4 = argv[4];
    // ofstream trackneuralpars(FileName4);
    // std::string const & FileName4 = argv[4];
    // ofstream HPofffreq(FileName4);


    // Set random number generator and seed
    RandomState rs;
    long seed=-time(0);
    rs.SetRandomSeed(seed);

    // Initialize Vector of base genomes because can't smoothly read from file
    TMatrix<double> basegenomematrix(1,Circuits,1,VectSize); //can't smoothly read from file, so store as matrix too

    // Randomly generate HPCTRNNs to find enough in the correct frequency band (((may eventually replace with evolution)))
    int i = 1;
    while (i <= Circuits){
      // Generate a random "genotype"
      TVector<double> genotype(1,VectSize);
      for (int i = 1; i <= VectSize; i++){
        genotype[i] = rs.UniformRandom(MinSearchValue,MaxSearchValue);
      }

      // Map from genotype to phenotype
      TVector<double> phenotype;
      phenotype.SetBounds(1, VectSize);
      GenPhenMapping(n, genotype, phenotype);

      // Run circuit and test frequency
     TVector<double> freq(1,n);
     frequencytest(phenotype,HPtausample,freq);

      if (abs(TargetFrequency-freq[1])<TargetMargin && abs(TargetFrequency-freq[2])<TargetMargin && abs(TargetFrequency-freq[3])<TargetMargin){
        for (int j=1; j<=VectSize; j++){
          basegenomematrix(i,j) = genotype[j];
          basegenomes << genotype[j] << " ";
        }
        //HPonfreq << freq[1] << " " << freq[2] << " " << freq[3] << endl;
        basegenomes << endl;
        cout << "Got " << i << " circuits in range" << endl;
        i ++;
      }
    }

    // For each HP timescale, repeat testing procedure
    for (double HPt = HPtaumin; HPt <= HPtaumax; HPt += .1) {

      cout<< "HPtau = " << HPt << endl; //progress statements

      for (int i=1; i<= Circuits; i++){
        TVector<double> genotype(1,VectSize);
        for (int j = 1; j <= VectSize; j++){
          genotype[j] = basegenomematrix(i,j);
        }

        // Map from genotype to phenotype
        TVector<double> phenotype;
        phenotype.SetBounds(1, VectSize);
        GenPhenMapping(n, genotype, phenotype);

        TVector<double> freq(1,n);
        frequencytest(phenotype,HPt,freq);

        HPonfreq << freq[1] << " " << freq[2] << " " << freq[3] << endl;

      }

    }
    return 0;
}

