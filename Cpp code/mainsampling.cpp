// ***************************************
// ALIFE 2023 Parameter Space Sampling Re-do 1/26/23
// ***************************************
// ENCODS Detail Par Space Sampling 5/26
// ~~~~~'s are for the 3-neuron case recording the neural parameter changes
// ^^^^^'s are for recording the neural outputs thorughout the timeseries
// ***************************************

#include "CTRNN.h"

// Global constants
const double TransientDuration = 500;  //in seconds, transient passed whenever HP is off
const double RunDuration = 50;
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

    // For each network size
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

      // Histogram of average dimensionalities observed AMONG THE OSCILLATING CIRCUITS (NEW-never fully implemented)
      TVector<double> mdimOG(1,maxNetworkSize);
      mdimOG.FillContents(0);
      TVector<double> mdimHPon(1,maxNetworkSize);
      mdimHPon.FillContents(0);
      TVector<double> mdimHPoff(1,maxNetworkSize);
      mdimHPoff.FillContents(0); 

      // Create (or read in) sample of (quasi-)random circuits:
      for (int m = 1; m <= Circuits; m += 1) {

        // Set up the circuit
        CTRNN c(n,WS,B,BT,WT,WR,BR);

        // Generate a random "genotype"
        TVector<double> genotype(1,VectSize);
        for (int i = 1; i <= VectSize; i++){
            genotype[i] = rs.UniformRandom(MinSearchValue,MaxSearchValue);
        }

          // Map from genotype to phenotype
      	TVector<double> phenotype;
      	phenotype.SetBounds(1, VectSize);
      	GenPhenMapping(n, genotype, phenotype);

          // Check whether the Center Crossing version will still be within [-16,16] bounds
          //
          //
          // keep regenerating until it is
          //
          //
        double thetastar; //Calculate the target center crossing bias for each neuron based on incoming(?is that what it is?) connection weights
        for (int i = 1; i<= n; i+= 1) {
              thetastar = 0;
              for (int j = 1; j <= n; j += 1) {
                thetastar += c.ConnectionWeight(i,j);
              }
            thetastar = thetastar / -2;
        }

        // For each circuit, repeat the experiment for several different initial conditions
        for (int r = 1; r <= Repetitions; r += 1) {

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
          // '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
          // 1/26 TOOK OUT ALL REFERENCES TO FITNESS - NOW ONLY SAMPLING FOR OSCILLATION AND ITS DIMENSION
          // '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
          // original oscillatory fitness
          // double OGoscfitness = 0;
 
          // original Dimension of oscillation
          int OGdim = 0;

          // oscillatory fitness with HP on
          // double HPonoscfitness = 0;

          // dimension of oscillation with HP on
          int HPondim = 0;

          // new oscillatory fitness with HP off
          // double newHPoffoscfitness = 0;

          // new dimension of oscillation with HP off
          int HPoffdim = 0;

          //Distance from center crossing target, given the weight values
          double CCdist = 0;

          // Initialize the state between [-5,5] at random
          c.RandomizeCircuitState(-5.0, 5.0, rs);  //does this really change the number if the seed is set at the beginning of the function?
          // cout << c.NeuronOutput(1) << "," << c.NeuronOutput(2) << "," << c.NeuronOutput(3) << endl;

          // Inactivate HP, just in case
          for (int i = 1; i<= n; i++){
            c.SetPlasticityBoundary(i, 0);
          }

          // Calculate CC distance at the start of the run for all n=3 circuits
          if (n==3){
            CCdist = 0;
            for (int i = 1; i<= n; i++) {
              CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
            }
            CCdist = sqrt(CCdist);
            centercrossingdistance << CCdist << " ";
          }


          // Run the circuit for the initial transient, tracking outputs for several n=3 circuits
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
//^^^^^^^^^^^^^^^^^^^^
            if (n==3&&m<250&&r==1) { //Only for size n=3, the first 250 circuits, on their first repetition
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
            }
            c.EulerStep(StepSize);
          }

          // Run the circuit to calculate its original dimension of oscillation
          TVector<double> pastNeuronOutput(1,n);
          TVector<double> activity(1,n);
          activity.FillContents(0.0);
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
            for (int i = 1; i <= n; i += 1) {
              pastNeuronOutput[i] = c.NeuronOutput(i);
            }
//^^^^^^^^^^^^^^^^^^^^^^
            if (n==3&&m<250&&r==1) {
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
            }
            c.EulerStep(StepSize);
            for (int i = 1; i <= n; i += 1) {
              activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
            }
          }
          // Calculate how many neurons demonstrate non-stationary activity
          int activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }

          //Only keep track of the highest dimension of oscillation throughout ICs
          if (activeneuroncounter > OGdim){
            OGdim = activeneuroncounter;
          }

// ----------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------

          // Turn on homeostatic plasticity mechanism 
          for (int i = 1; i<= n; i++){
            c.SetPlasticityBoundary(i, 0.25);
          }


           // Run for ransient, tracking neural pars for several n=3 circuits
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (n==3&&m<250&&r==1){
              trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronTimeConstant(3) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.NeuronBias(3) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(1,3) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << " " << c.ConnectionWeight(2,3) << " " << c.ConnectionWeight(3,1) << " " << c.ConnectionWeight(3,2) << " " << c.ConnectionWeight(3,3) << endl;
              //trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << endl;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
            }
            c.EulerStep(StepSize);
            
          }


          // Run the circuit to calculate its HP on dimension of oscillation
          activity.FillContents(0.0);
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
            for (int i = 1; i <= n; i += 1) {
              pastNeuronOutput[i] = c.NeuronOutput(i);
            }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~~~~~~~~~~
            if (n==3&&m<250&&r==1){
              trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronTimeConstant(3) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.NeuronBias(3) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(1,3) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << " " << c.ConnectionWeight(2,3) << " " << c.ConnectionWeight(3,1) << " " << c.ConnectionWeight(3,2) << " " << c.ConnectionWeight(3,3) << endl;
              //trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
            }
            c.EulerStep(StepSize);
            for (int i = 1; i <= n; i += 1) {
              activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
            }
          }

          // Calculate how many neurons demonstrate non-stationary activity
          activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }

          //Only keep track of the highest dimension of oscillation throughout ICs               **Might consider checking the activity level of the neural parameters, since they are dimensions of possible oscillation (the point)
          if (activeneuroncounter > HPondim){
            HPondim = activeneuroncounter;
          }

// --------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------

          //Turn HP back off
          for (int i = 1; i<= n; i++){
            c.SetPlasticityBoundary(i, 0);
          }

          //Calculate post-HP CC distance
          if (n==3){
          CCdist = 0;
          for (int i = 1; i<= n; i+= 1) {
            CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
          }
          CCdist = sqrt(CCdist);
          centercrossingdistance << CCdist << endl;
          }
          
          //Pass transient
          for (double time = StepSize; time <= TransientDuration; time += StepSize) {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if (n==3&&m<250&&r==1){
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
            }
            c.EulerStep(StepSize);
          }

          // Run the circuit to calculate its new HP off oscillatory fitness and dimension of oscillation
          activity.FillContents(0.0);
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
            for (int i = 1; i <= n; i += 1) {
              pastNeuronOutput[i] = c.NeuronOutput(i);
            }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if (n==3&&m<250&&r==1){  
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
            }
            c.EulerStep(StepSize);
            for (int i = 1; i <= n; i += 1) {
              activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
            }
          }

          // Calculate how many neurons demonstrate non-stationary activity
          activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }

          //Only keep track of the highest dimension of oscillation throughout ICs
          if (activeneuroncounter > HPoffdim){
            HPoffdim = activeneuroncounter;
          }
        
      
    
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
        OGdimension << OGdim << endl;
        trackneuralpars << endl;
        HPondimension << HPondim << endl;
        HPoffdimension << HPoffdim << endl;


        //cout << m << " " << OGoscfitness << " " << OGdim << " " << HPonoscfitness << " " << HPondim << " " << newHPoffoscfitness << " " << newHPoffdim << endl;

        }
      }
    }
    OGdimension.close();
    trackneuralpars.close();
    HPondimension.close();
    HPoffdimension.close();
    neuralstates.close();
    centercrossingdistance.close();

    // Finished
    return 0;
}
