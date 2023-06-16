// ***************************************
// Replicating Randy's Fig 11 from 2006
// Alternative version (not used) with all one stream (OG to HP on to HPoff in one circuit)
// ***************************************

#include "CTRNN.h"

// Global constants
const double TransientDuration = 500;
const double RunDuration = 50;
const double StepSize = 0.1;
const double maxNetworkSize = 20;
const double minNetworkSize = 1;
const double WR = 16.0;
const double BR = 16.0;
const double TMIN = 0.5;
const double TMAX = 10.0;
const double Circuits = 10000; //10^4
const double Repetitions = 10;

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
    std::string const & mdimOGFileName = argv[2];
    ofstream mdimfileOG(mdimOGFileName);
    std::string const & mdimHPonFileName = argv[3];
    ofstream mdimfileHPon(mdimHPonFileName);    
    std::string const & mdimHPoffFileName = argv[4];
    ofstream mdimfileHPoff(mdimHPoffFileName);

    // Set random number generator and seed
    RandomState rs;
    long seed=-time(0);
    rs.SetRandomSeed(seed);

		// Plasticity parameters
		int WS = 1; 				// Window Size of Plastic Rule (in steps size) (so 1 is no window)
		double B = 0.25; 		// Plasticity Low Boundary
		double BT = 20.0;		// Bias Time Constant
		double WT = 40.0;		// Weight Time Constant

    // For each neuron size
    for (int n = minNetworkSize; n <= maxNetworkSize; n += 1) {
		//for (int n = 30; n <= 100; n += 10) {

      // Number of parameters
      int VectSize = n*n + 2*n;

      // Number of circuits that "oscillate"
      int counterOG = 0;
      int counterHPon = 0;
      int counterHPoff = 0;
      int counterICmatters = 0;

      // Histogram of dimensionalities observed
      TVector<int> mdimOG(1,20);
      mdimOG.FillContents(0);
      TVector<int> mdimHPon(1,20);
      mdimHPon.FillContents(0);
      TVector<int> mdimHPoff(1,20);
      mdimHPoff.FillContents(0);
      
      // Create 10^5 random circuits:
      for (int m = 1; m <= Circuits; m += 1) {

        // Max Dim observed for this circuits
        int maxdimactOG = 0;
        int maxdimactHPon = 0;
        int maxdimactHPoff = 0;

        // Set up the circuit
        CTRNN c(n,WS,0,BT,WT,WR,10000);

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

        // c.SetCenterCrossing();

        // For each circuit, repeat the experiment 10 times
        for (int r = 1; r <= Repetitions; r += 1) {

          // Initialize the state between [-16,16] at random
          c.RandomizeCircuitState(-16.0, 16.0, rs);

          // Run the circuit for the initial transient
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
              c.EulerStep(StepSize);
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
          if (activeneuroncounter > maxdimactOG){
            maxdimactOG = activeneuroncounter;
          }

          //Turn HP on
          for (int i=1;i<=n;i++){
            c.SetPlasticityBoundary(i,B);
          }

          // Run the circuit for the initial transient
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
              c.EulerStep(StepSize);
          }

          // Run the circuit to calculate whether it's oscillating or not
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
          activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }
          if (activeneuroncounter > maxdimactHPon){
            maxdimactHPon = activeneuroncounter;
          }

          //Turn HP back off
          for (int i=1;i<=n;i++){
            c.SetPlasticityBoundary(i,0);
          }

          // Run the circuit for the initial transient
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
              c.EulerStep(StepSize);
          }

          // Run the circuit to calculate whether it's oscillating or not
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
          activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }
          if (activeneuroncounter > maxdimactHPoff){
            maxdimactHPoff = activeneuroncounter;
          }
        }
        
          if (maxdimactOG > 0){
            counterOG++;
            mdimOG[maxdimactOG] = mdimOG[maxdimactOG] + 1;
          }

          if (maxdimactHPon > 0){
            counterHPon++;
            mdimHPon[maxdimactHPon] = mdimHPon[maxdimactHPon] + 1;
          }

          if (maxdimactHPoff > 0){
            counterHPoff++;
            mdimHPoff[maxdimactHPoff] = mdimHPoff[maxdimactHPoff] + 1;
          }

        

      }

      double percentageOG = 100 * (counterOG / Circuits);
      double percentageHPon = 100 * (counterHPon / Circuits);
      double percentageHPoff = 100 * (counterHPoff / Circuits);
      outputfile << n << " " << percentageOG << " " << percentageHPon << " " << percentageHPoff << endl;
      cout << n << " " << percentageOG << ", " << percentageHPon << ", " << percentageHPoff << endl;

      for (int i = 1; i <= 20; i += 1) {
        mdimfileOG << 100 * (mdimOG[i] / Circuits) << " ";
        mdimfileHPon << 100 * (mdimHPon[i] / Circuits) << " ";
        mdimfileHPoff << 100 * (mdimHPoff[i] / Circuits) << " ";
      }
      mdimfileOG << endl;
      mdimfileHPon << endl;
      mdimfileHPoff << endl;

    }

    outputfile.close();
    mdimfileOG.close();
    mdimfileHPon.close();
    mdimfileHPoff.close();

    // Finished
    return 0;
}
