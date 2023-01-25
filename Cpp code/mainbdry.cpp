// ***************************************
// ENCODS Detail Par Space Sampling 5/26
// ~~~~~'s are for the 3-neuron case recording the neural parameter changes
// ^^^^^'s are for recording the neural outputs thorughout the timeseries
// ***************************************

#include "CTRNN.h"

// Global constants
const double TransientDuration1 = 300;  //in seconds, transient passed whenever HP is off
const double TransientDuration2 = 600; //transient passed whenever HP is on
const double RunDuration = 50;
const double StepSize = 0.1;
//const double maxNetworkSize = 20;
//const double minNetworkSize = 1;
const double minbdry = 0;
const double maxbdry = .5;
const double WR = 16.0;          //Weight range
const double BR = 16.0;          //Bias Range
const double TMIN = 0.5;
const double TMAX = 10.0;
const double Circuits = 5000;     //How many circuits in sample
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
    ofstream OGHPofffitness(FileName1);
    std::string const & FileName2 = argv[2];
    ofstream OGHPoffdimension(FileName2);
    std::string const & FileName3 = argv[3];
    ofstream trackneuralpars(FileName3);
    std::string const & FileName4 = argv[4];
    ofstream HPonfitness(FileName4);
    std::string const & FileName5 = argv[5];
    ofstream HPondimension(FileName5);
    std::string const & FileName6 = argv[6];
    ofstream newHPofffitness(FileName6);
    std::string const & FileName7 = argv[7];
    ofstream newHPoffdimension(FileName7);
    std::string const & FileName8 = argv[8];
    ofstream neuralstates(FileName8);
    std::string const & FileName9 = argv[9];
    ofstream centercrossingdistance(FileName9);

    // Set random number generator and seed
    RandomState rs;
    long seed=-time(0);
    rs.SetRandomSeed(seed);

		// Plasticity parameters
		int WS = 1; 				// Window Size of Plastic Rule (in steps size) (so 1 is no window)
		//double B = 0.25; 		// Plasticity Low Boundary
    int n = 2; //circuit size = 2
		double BT = 20.0;		// Bias Time Constant
		double WT = 40.0;		// Weight Time Constant

    // For each boundray size
    for (double B = minbdry; B <= maxbdry; B += 0.05) {

      // Number of parameters
      int VectSize = n*n + 2*n;

      // Number of circuits that "oscillate"
      int counter = 0;

      // Number of circuits that "oscillate" after HP turned off
      int counter2 = 0;

      // Histogram of dimensionalities observed
      //TVector<int> mdim(1,20);
      //mdim.FillContents(0);
      //TVector<int> mdim2(1,20);
      //mdim2.FillContents(0);

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

        // For each circuit, repeat the experiment for R different initial conditions
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

          // original oscillatory fitness
          double OGoscfitness = 0;
 
          // original Dimension of oscillation
          int OGdim = 0;

          // oscillatory fitness with HP on
          double HPonoscfitness = 0;

          // dimension of oscillation with HP on
          int HPondim = 0;

          // new oscillatory fitness with HP off
          double newHPoffoscfitness = 0;

          // new dimension of oscillation with HP off
          int newHPoffdim = 0;

          //Center crossing target for each neuron
          double thetastar = 0;

          //Distance from center crossing target, given the weight values
          double CCdist = 0;

          // Initialize the state between [-16,16] at random
          c.RandomizeCircuitState(-16.0, 16.0, rs);

          // Inactivate HP
          for (int i = 1; i<= n; i++){
            c.ChangeHPBoundary(i, 0);
          }

          // Run the circuit for the initial transient
          for (double time = StepSize; time <= TransientDuration1; time += StepSize) {
//^^^^^^^^^^^^^^^^^^^^
            if (n==2) {
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
                thetastar = thetastar / -2;
                CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
          }

          // Run the circuit to calculate its original oscillatory fitness and dimension of oscillation
          TVector<double> pastNeuronOutput(1,n);
          TVector<double> activity(1,n);
          activity.FillContents(0.0);
          double oscfitness = 0;
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
              for (int i = 1; i <= n; i += 1) {
                pastNeuronOutput[i] = c.NeuronOutput(i);
              }
//^^^^^^^^^^^^^^^^^^^^^^
            if (n==2) {
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
              thetastar = thetastar / -2;
              CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
              double normvelvector = 0;
              for (int i = 1; i <= n; i += 1) {
                activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
                normvelvector += (c.NeuronOutput(i) - pastNeuronOutput[i])*(c.NeuronOutput(i) - pastNeuronOutput[i]);
              }
              oscfitness += sqrt(normvelvector);
          }

          oscfitness = oscfitness/(RunDuration/StepSize);

          //Only keep track of the highest oscillatory fitness throughout ICs
          if (oscfitness > OGoscfitness){
              OGoscfitness = oscfitness;
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
            c.ChangeHPBoundary(i, 0.25);
          }


           // Run for long transient, keeping track of neural pars
          for (double time = StepSize; time <= TransientDuration2; time += StepSize) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (n==2){
              //trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronTimeConstant(3) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.NeuronBias(3) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(1,3) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << " " << c.ConnectionWeight(2,3) << " " << c.ConnectionWeight(3,1) << " " << c.ConnectionWeight(3,2) << " " << c.ConnectionWeight(3,3) << endl;
              trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << endl;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
                thetastar = thetastar / -2;
                CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
            
          }


          // Run the circuit to calculate its HP on oscillatory fitness and dimension of oscillation
          activity.FillContents(0.0);
          oscfitness = 0;
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
            for (int i = 1; i <= n; i += 1) {
              pastNeuronOutput[i] = c.NeuronOutput(i);
            }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~~~~~~~~~~
            if (n==2){
              //trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronTimeConstant(3) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.NeuronBias(3) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(1,3) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << " " << c.ConnectionWeight(2,3) << " " << c.ConnectionWeight(3,1) << " " << c.ConnectionWeight(3,2) << " " << c.ConnectionWeight(3,3) << endl;
              trackneuralpars << c.NeuronTimeConstant(1) << " " << c.NeuronTimeConstant(2) << " " << c.NeuronBias(1) << " " << c.NeuronBias(2) << " " << c.ConnectionWeight(1,1) << " " << c.ConnectionWeight(1,2) << " " << c.ConnectionWeight(2,1) << " " << c.ConnectionWeight(2,2) << endl;
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
                thetastar = thetastar / -2;
                CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
            double normvelvector = 0;
            for (int i = 1; i <= n; i += 1) {
              activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
              normvelvector += (c.NeuronOutput(i) - pastNeuronOutput[i])*(c.NeuronOutput(i) - pastNeuronOutput[i]);
            }
            oscfitness += sqrt(normvelvector);
          }

          oscfitness = oscfitness/(RunDuration/StepSize);

          //Only keep track of the highest oscillatory fitness throughout ICs
          if (oscfitness > HPonoscfitness){
              HPonoscfitness = oscfitness;
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

          //Turn HP off
          for (int i = 1; i<= n; i++){
            c.ChangeHPBoundary(i, 0);
          }
          
          //Pass short transient
          for (double time = StepSize; time <= TransientDuration1; time += StepSize) {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if (n==2){
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
                thetastar = thetastar / -2;
                CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
          }

          // Run the circuit to calculate its new HP off oscillatory fitness and dimension of oscillation
          activity.FillContents(0.0);
          oscfitness = 0;
          for (double time = StepSize; time <= RunDuration; time += StepSize) {
            for (int i = 1; i <= n; i += 1) {
              pastNeuronOutput[i] = c.NeuronOutput(i);
            }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if (n==2){  
              //neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << " " << c.NeuronOutput(3) << endl;
              neuralstates << c.NeuronOutput(1) << " " << c.NeuronOutput(2) << endl;
              CCdist = 0;
              for (int i = 1; i<= n; i+= 1) {
                thetastar = 0;
                for (int j = 1; j <= n; j += 1) {
                  thetastar += c.ConnectionWeight(i,j);
                }
                thetastar = thetastar / -2;
                CCdist += (c.NeuronBias(i) - thetastar) * (c.NeuronBias(i) - thetastar);
              }
              CCdist = sqrt(CCdist);
              centercrossingdistance << CCdist << endl;
            }
            c.EulerStep(StepSize);
            double normvelvector = 0;
            for (int i = 1; i <= n; i += 1) {
              activity[i] += fabs(c.NeuronOutput(i) - pastNeuronOutput[i]);
              normvelvector += (c.NeuronOutput(i) - pastNeuronOutput[i])*(c.NeuronOutput(i) - pastNeuronOutput[i]);
            }
            oscfitness += sqrt(normvelvector);
          }

          oscfitness = oscfitness/(RunDuration/StepSize);

          //Only keep track of the highest oscillatory fitness throughout ICs
          if (oscfitness > newHPoffoscfitness){
              newHPoffoscfitness = oscfitness;
          }

          // Calculate how many neurons demonstrate non-stationary activity
          activeneuroncounter = 0;
          for (int i = 1; i <= n; i += 1) {
            if (activity[i] > 0.05){
                activeneuroncounter++;
            }
          }

          //Only keep track of the highest dimension of oscillation throughout ICs
          if (activeneuroncounter > newHPoffdim){
            newHPoffdim = activeneuroncounter;
          }
        
      
    
// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------

        OGHPofffitness << OGoscfitness << endl;
        OGHPoffdimension << OGdim << endl;
        trackneuralpars << " " << endl;
        HPonfitness << HPonoscfitness << endl;
        HPondimension << HPondim << endl;
        newHPofffitness << newHPoffoscfitness << endl;
        newHPoffdimension << newHPoffdim << endl;


        //cout << m << " " << OGoscfitness << " " << OGdim << " " << HPonoscfitness << " " << HPondim << " " << newHPoffoscfitness << " " << newHPoffdim << endl;

        }
      }
    }
    OGHPofffitness.close();
    OGHPoffdimension.close();
    trackneuralpars.close();
    HPonfitness.close();
    HPondimension.close();
    newHPofffitness.close();
    newHPoffdimension.close();
    neuralstates.close();
    centercrossingdistance.close();

    // Finished
    return 0;
}
