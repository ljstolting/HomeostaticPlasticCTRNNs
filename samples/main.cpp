// ***************************************
// Replicating Randy's Fig 11 from 2006
// ***************************************

#include "CTRNN.h"
#include <stdio.h>

// Global constants
const double TransientDuration = 500;
const double RunDuration = 50;
const double StepSize = 0.1;
const double WR = 16.0;
const double BR = 16.0;
const double TMIN = 0.5;
const double TMAX = 10.0;
const double Circuits = 10000; //10^4
const double Repetitions = 10;

const double HPPlasticBoundary = 0.25;
const int stages = 1; //how many stagest of transient of length specified above
int HPonwhen[stages];


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
    std::string const & FileName = argv[1];
    ofstream outputfile(FileName);
    std::string const & mdimFileName = argv[2];
    ofstream mdimfile(mdimFileName);
    std::string const & trackneuralstatesfile = argv[3];
    ofstream trackneuralstates(trackneuralstatesfile);

    // Set random number generator and seed
    RandomState rs;
    long seed=-time(0);
    rs.SetRandomSeed(seed);

		// Plasticity parameters
		int WS = 1; 				// Window Size of Plastic Rule (in steps size) (so 1 is no window)
		double BT = 20.0;		// Bias Time Constant
		double WT = 40.0;		// Weight Time Constant
    HPonwhen[0] = 0;    // Set whether HP is on or off during each transient stage

    // For each neuron size
    for (int n = 1; n <= 20; n += 1) {
		//for (int n = 30; n <= 100; n += 10) {

      // Number of parameters
      int VectSize = n*n + 2*n;

      // Number of circuits that "oscillate"
      int counter = 0;

      // Histogram of dimensionalities observed
      TVector<int> mdim(1,20);
      mdim.FillContents(0);

      int ICdependentcounter = 0;

      // Create 10^5 random circuits:
      for (int m = 1; m <= Circuits; m += 1) {

        // Max Dim observed for this circuits
        int maxdimact = 0;
        int formerdimact = 0; //for seeing if the dimension ever changes throughout the initial conditions

        // Set up the circuit
        CTRNN c(n,WS,HPPlasticBoundary,BT,WT,WR,10000);

        // Generate a random "genotype"
        TVector<double> genotype(1,VectSize);
        for (int i = 1; i <= VectSize; i++){
            genotype[i] = rs.UniformRandom(MinSearchValue,MaxSearchValue);
        }

        // Map from genotype to phenotype
      	TVector<double> phenotype;
      	phenotype.SetBounds(1, VectSize);
      	GenPhenMapping(n, genotype, phenotype);

        // Set phenotype to circuit
        int k = 1;
        // Time-constants
        for (int i = 1; i <= n; i++) {
          c.SetNeuronTimeConstant(i, phenotype(k));
          k++;
        }
        // Bias
        for (int i = 1; i <= n; i++) {
          c.SetNeuronBias(i, phenotype(k));
          k++;
        }
        // Weights
        for (int i = 1; i <= n; i++) {
          for (int j = 1; j <= n; j++) {
            c.SetConnectionWeight(i, j, phenotype(k));
            k++;
          }
        }

        //c.SetCenterCrossing();

        bool ICdependentindicator = false;

        // For each circuit, repeat the experiment 10 times
        for (int r = 1; r <= Repetitions; r += 1) {

          // Initialize the state between [-16,16] at random
          c.RandomizeCircuitState(-16.0, 16.0, rs);

          // Run the circuit for the initial transient FOR HOWEVER MANY STAGES WITH HP ON OR OFF
          for (int stage=0;stage<stages;stage++){
            // Turn HP on or off according to specification
            for (int i=1;i<=n;i++){
                c.SetPlasticityBoundary(i,HPonwhen[stage]*HPPlasticBoundary);
                //cout<<c.PlasticityBoundary(i);
            }
            for (double time = StepSize; time <= TransientDuration; time += StepSize) {
                c.EulerStep(StepSize);
                //trackneuralstates << c.NeuronOutput(1) << endl;
            }
          }

          // Run the circuit to calculate whether it's oscillating or not
          TVector<double> pastNeuronOutput(1,n);
          TVector<double> activity(1,n);
          activity.FillContents(0.0);
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
              for (int i = 1; i <= n; i += 1) {
                pastNeuronOutput[i] = c.NeuronOutput(i);
              }
              c.EulerStep(StepSize);
              for (int i = 1; i <= n; i += 1) {
                activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
              }
          }

          // Keep track of how many neurons demonstrate non-stationary activity
          int activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }
          if ((r!=1) && (formerdimact+activeneuroncounter!=0) && (formerdimact*activeneuroncounter==0)){ //if exactly one of them is zero
            //cout << "formerly:" << formerdimact << " now:" << activeneuroncounter << endl;  
            ICdependentindicator = true;
            //cout << "circuit #" << m << "repetition #" << r << "vs." << r-1 << endl;
          }
          if (activeneuroncounter > maxdimact){
            maxdimact = activeneuroncounter;
          }
          //cout << r << " " << formerdimact << " ";
          formerdimact = activeneuroncounter;
        }
        //cout << endl;

        if (ICdependentindicator){
          ICdependentcounter ++;
        }

        if (maxdimact > 0){
          counter++;
          mdim[maxdimact] = mdim[maxdimact] + 1;
        }

      }

      double percentage = 100 * (counter / Circuits);
      outputfile << n << " " << percentage << " " << ICdependentcounter << endl;
      cout << n << " " << percentage << " " << ICdependentcounter << endl;

      for (int i = 1; i <= 20; i += 1) {
        mdimfile << 100 * (mdim[i] / Circuits) << " ";
      }
      mdimfile << endl;

    }

    outputfile.close();
    mdimfile.close();
    trackneuralstates.close();

    // Finished
    return 0;
}
