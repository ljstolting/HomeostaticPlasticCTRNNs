#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double TransientDuration = 500; //this is a proper transient for up to about tau=40, but for greater it should probably be longer?
const double BehaviorDuration = 500;
//const double OscillationDuration = 100;
const double DurThreshold = 1000;
const double StepSize = 0.01;
const double TargetFrequency = .05;
const double DistThreshold = 0.075;

// EA params
const int POPSIZE = 50;
const int GENS = 50;
const double MUTVAR = 0.01;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Nervous system params
const int N = 3;
const double WR = 16.0; 
const double BR = 16.0; 
const double TMIN = 1; 
const double TMAX = 1; 

// Plasticity parameters
const int WS = 1;			// Window Size of Plastic Rule (in steps size) (so 1 is no window)
const double B = 0.25; 		// Plasticity Low Boundary
double BT = 1.0;	// Bias Time Constant
double WT = 1.0;	// Weight Time Constant

// Tau Sweep parameters
const double HPtaustart = 1;  //may later include data for .5 as interest (faster than neural timescale)
const double HPtaustop = 100;
const double HPtaustep = 1;
const int Circuits = 25; //Number of circuits to evolve (runs to complete) at each value of HP tau
const int Repetitions = 1; 
//^ Number of initial conditions from which to test each circuit (Does HP only sometimes find a limit cycle? does turning it off only sometimes affect performance?)
//const bool doBehavior = 0; //Whether to run the best circuits to record their outputs and parameters for BehaviorDuration

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}
// ------------------------------------
// Fitness function
// ------------------------------------
double FreqCalcHPon(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TVector<double> pastNeuronOutput(1,N);
	TVector<double> CumRateChange(1,N);

	// Create the agent
	CTRNN Agent;

	// Instantiate the nervous system
	Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
	//cout << BT << "," << WT << endl;
	
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Agent.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}

	// Initialize the state at an output of 0.5 for all neurons in the circuit
	Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Run the circuit for an initial transient (Fitness is not evaluated, HP is on)
	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
	}

	// Run the circuit to calculate whether the neurons are oscillating or not (according to Beer 2006)(HP may be on or off)
	CumRateChange.FillContents(0.0);
	for (double time = StepSize; time <= 50; time += StepSize) {
		for (int i = 1; i <= N; i += 1) {
			pastNeuronOutput[i] = Agent.NeuronOutput(i);
		}
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i]));
		}
	}
	int OscillationFlag = 0;
	for (int i = 1; i <= N; i += 1) {
		if (CumRateChange[i]> 0.05)
		{
			OscillationFlag = 1;
		}
	}

	// Only continue fitness if at least one neuron oscillating
	if (OscillationFlag == 1)
	{
		//cout << "B = " << PlasticBoundary << " all oscillating" << endl;
		// Run the circuit to calculate the frequency of oscillation
		// 1. Record the current N^2 + 2N-dimensional state (weights,biases, and neural states)
		int k = 1;
		TVector<double> goalState(1,VectSize);
		for (int i = 1; i <= N; i += 1){
			// goalState[i] = Agent.NeuronOutput(i);
			goalState[k] = Agent.NeuronState(i);
			k ++;
		}
		for (int i = 1; i <= N; i ++){ //record biases
			goalState[k] = Agent.NeuronBias(i);
			k++;
		}

		for (int i = 1; i <= N; i++) { //record weights
			for (int j = 1; j <= N; j++) {
				goalState[k] = Agent.ConnectionWeight(i,j);
				k++;
			}
		}
		// 2. Integrate the system until it's far enough from the starting state
		double dist = 0.0;
		double time = 0.0;
		while  ((dist < DistThreshold) && (time < DurThreshold))
		{
			Agent.EulerStep(StepSize);
			// Re-calculate Euclidean distance with state
			k = 1;
			dist = 0;
			for (int i = 1; i <= N; i += 1) 
			{
				// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
				dist += pow(goalState(k) - Agent.NeuronState(i), 2);
				k++;
			}
			for (int i = 1; i <= N; i ++){ //factor in biases
				dist += pow(goalState(k) - Agent.NeuronBias(i),2);
				k++;
			}

			for (int i = 1; i <= N; i++) { //factor in weights
				for (int j = 1; j <= N; j++) {
					dist += pow(goalState(k) - Agent.ConnectionWeight(i,j),2);
					k++;
				}
			}
			dist = sqrt(dist);
			// Update time
			time += StepSize;
		}	

		// If it left in a decent time (meaning it was truly oscillating), then keep going
		if (time < DurThreshold)
		{
			// 3. Integrate the system until it's close enough again! (or until a reasonable length of time runs out)
			time = 0.0;
			while ((dist >= DistThreshold) && (time < DurThreshold))
			{
				Agent.EulerStep(StepSize);
				// Re-calculate Euclidean distance with state
				dist = 0;
				k = 1;
				for (int i = 1; i <= N; i += 1) 
				{
				// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
				dist += pow(goalState(k) - Agent.NeuronState(i), 2);
				k++;
				}
				for (int i = 1; i <= N; i ++){ //factor in biases
					dist += pow(goalState(k) - Agent.NeuronBias(i),2);
					k++;
				}

				for (int i = 1; i <= N; i++) { //factor in weights
					for (int j = 1; j <= N; j++) {
						dist += pow(goalState(k) - Agent.ConnectionWeight(i,j),2);
						k++;
					}
				}
				dist = sqrt(dist);
				// Update time
				time += StepSize;
			}	
			if (time < DurThreshold)
			{
				double measuredFrequency = 1/time;
				//cout << "B = " << PlasticBoundary << " freq measured = " << measuredFrequency << endl;
				return measuredFrequency;
			}
			else
			{
				//cout << "Left initial state but didn't find full cycle" << endl;
				return 0.0;
			}
		}
		else
		{
			//cout << "oscillation detected but did not leave initial state in reasonable time" << endl;
			return 0.0;
		}		
	}
	else
	{
		//cout << "none oscillating" << endl;
		return 0.0;
	}
}

double FitnessFunctionHPon(TVector<double> &genotype, RandomState &rs){
	double freq = FreqCalcHPon(genotype,rs);
	return 1.0 - ((1.0/pow(TargetFrequency,2.0))*pow((freq-TargetFrequency),2.0));
}

double FreqCalcHPoff(TVector<double> &genotype, RandomState &rs)
//note that now the system is autonomous wrt neural pars, so I don't need to calculate distance in those dimensions (they would always be zero)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	TVector<double> pastNeuronOutput(1,N);
	TVector<double> CumRateChange(1,N);

	// Create the agent
	CTRNN Agent;

	// Instantiate the nervous system
	Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
	//cout << BT << "," << WT << endl;
	
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Agent.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}

	// Initialize the state at an output of 0.5 for all neurons in the circuit
	Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Run the circuit for an initial transient (Fitness is not evaluated, HP is on)
	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
	}

	for (int i=1;i<=N;i++){
		Agent.SetPlasticityBoundary(i,0); //turn HP off
	}

	// Pass another transient after HP is off 
	for (double time = StepSize; time <= TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
	}

	// Run the circuit to calculate whether the neurons are oscillating or not (according to Beer 2006)(HP may be on or off)
	CumRateChange.FillContents(0.0);
	for (double time = StepSize; time <= 50; time += StepSize) {
		for (int i = 1; i <= N; i += 1) {
			pastNeuronOutput[i] = Agent.NeuronOutput(i);
		}
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			CumRateChange[i] += abs((Agent.NeuronOutput(i) - pastNeuronOutput[i]));
		}
	}
	int OscillationFlag = 0;
	for (int i = 1; i <= N; i += 1) {
		if (CumRateChange[i]> 0.05)
		{
			OscillationFlag = 1;
		}
	}

	// Only continue fitness if at least one neuron oscillating
	if (OscillationFlag == 1)
	{
		//cout << "B = " << PlasticBoundary << " all oscillating" << endl;
		// Run the circuit to calculate the frequency of oscillation
		// 1. Record the current N-dimensional state
		TVector<double> goalState(1,N);
		for (int i = 1; i <= N; i += 1) 
		{
			// goalState[i] = Agent.NeuronOutput(i);
			goalState[i] = Agent.NeuronState(i);
		}
		// 2. Integrate the system until it's far enough from the starting state
		double dist = 0.0;
		double time = 0.0;
		while  ((dist < DistThreshold) && (time < DurThreshold))
		{
			Agent.EulerStep(StepSize);
			// Re-calculate Euclidean distance with state
			dist = 0;
			for (int i = 1; i <= N; i += 1) 
			{
				// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
				dist += pow(goalState(i) - Agent.NeuronState(i), 2);
			}
			dist = sqrt(dist);
			// Update time
			time += StepSize;
		}	

		// If it left in a decent time (meaning it was truly oscillating), then keep going
		if (time < DurThreshold)
		{
			// 3. Integrate the system until it's close enough again! (or until a reasonable length of time runs out)
			time = 0.0;
			while ((dist >= DistThreshold) && (time < DurThreshold))
			{
				Agent.EulerStep(StepSize);
				// Re-calculate Euclidean distance with state
				dist = 0;
				for (int i = 1; i <= N; i += 1) 
				{
					// dist += pow(goalState(i) - Agent.NeuronOutput(i), 2);
					dist += pow(goalState(i) - Agent.NeuronState(i), 2);					
				}
				dist = sqrt(dist);
				// Update time
				time += StepSize;
			}	
			if (time < DurThreshold)
			{
				double measuredFrequency = 1/time;
				//cout << "B = " << PlasticBoundary << " freq measured = " << measuredFrequency << endl;
				return measuredFrequency;
			}
			else
			{
				//cout << "Left initial state but didn't find full cycle" << endl;
				return 0.0;
			}
		}
		else
		{
			//cout << "oscillation detected but did not leave initial state in reasonable time" << endl;
			return 0.0;
		}		
	}
	else
	{
		//cout << "none oscillating" << endl;
		return 0.0;
	}
}

// ------------------------------------
// Behavior
// ------------------------------------
void Behavior(TVector<double> &genotype)
{
	RandomState rs;
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	ofstream nfile("neural.dat");
	ofstream wfile("weights.dat");
	ofstream bfile("biases.dat");

	// Create the agent
	CTRNN Agent;

	// Instantiate the nervous system
	Agent.SetCircuitSize(N,WS,B,BT,WT,WR,BR);
	//cout << BT << "," << WT << endl;
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Agent.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
	}
		
	// Initialize the state at an output of 0.5 for all neurons in the circuit
	Agent.RandomizeCircuitOutput(0.5, 0.5);

	// Run the circuit, recording the neural parameters with HP on
	for (double time = StepSize; time < TransientDuration; time += StepSize) {
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			nfile << Agent.NeuronOutput(i) << " ";
		}
		nfile << endl;
		for (int i = 1; i <= N; i += 1) {
			bfile << Agent.NeuronBias(i) << " ";
			for (int j = 1; j <= N; j += 1) {
				wfile << Agent.ConnectionWeight(i,j) << " ";
			}
		}
		bfile << endl;
		wfile << endl;
	}

	for (int i=1;i<=N;i++){
		Agent.SetPlasticityBoundary(i,0); //turn off HP
	}
	//Run the circuit, recording parameters, with HP off    THIS MIGHT HAVE BEEN A MISTAKE ADD
	for (double time = StepSize; time < BehaviorDuration/2; time += StepSize) {
		Agent.EulerStep(StepSize);
		for (int i = 1; i <= N; i += 1) {
			nfile << Agent.NeuronOutput(i) << " ";
		}
		nfile << endl;
		for (int i = 1; i <= N; i += 1) {
			bfile << Agent.NeuronBias(i) << " ";
			for (int j = 1; j <= N; j += 1) {
				wfile << Agent.ConnectionWeight(i,j) << " ";
			}
		}
		bfile << endl;
		wfile << endl;
	}

	nfile.close();
	bfile.close();
	wfile.close();

}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf > 0.99) {
		cout << "phase 1 complete" << endl;
		return 1;
	}
	else return 0;
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ------------------------------------
// Evolutionary Search Program
// ------------------------------------
double EvolRun(long IDUM){
	// Configure the search

	TSearch s(VectSize);

	s.SetRandomSeed(IDUM);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(0); 

	/* Stage 1 */
	// s.SetSearchTerminationFunction(TerminationFunction);
	// s.SetEvaluationFunction(FitnessFunction1); 
	// s.ExecuteSearch();
	/* Stage 2 */
	s.SetSearchTerminationFunction(NULL);
	s.SetEvaluationFunction(FitnessFunctionHPon);
	s.ExecuteSearch();

	return s.BestPerformance();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
	RandomState rs;

	#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
	#endif

	ofstream OGfreqfile;
	OGfreqfile.open("OGfrequency.dat");
	ofstream HPofffreqfile;
	HPofffreqfile.open("HPofffrequency.dat");
	ofstream EvolvedPhenFile;
	EvolvedPhenFile.open("EvolvedPhenotypes.dat");

	double attainedfitness = 0;
	int attempts = 0;

	for (double HPtau = HPtaustart;HPtau<=HPtaustop;HPtau+=HPtaustep){
		BT = HPtau;
		WT = HPtau;
		for (int c=1;c<=Circuits;c++){
			long IDUM=-time(0);
			attainedfitness = 0;
			attempts = 0;
			while (attainedfitness < 0.999996 && attempts < 10){  //make sure that every circuit attains a reasonable fitness (good target freq)
				attempts ++;
				attainedfitness = EvolRun(IDUM);
				if (attempts == 10){cout << "warning, max of 10 attempts reached" << endl;}
			}
			ifstream genefile("best.gen.dat");
			TVector<double> genotype(1, VectSize);
			TVector<double> phenotype(1,VectSize);
			genefile >> genotype;
			GenPhenMapping(genotype,phenotype);
			//EvolvedPhenFile << phenotype << endl;
			//if((HPtau==1. || HPtau==25. || HPtau==50. || HPtau==100.) && c==1){Behavior(genotype);} //record the timeseries of select circuits from sweep
			for (int r=1;r<=Repetitions;r++){
				OGfreqfile << FreqCalcHPon(genotype,rs) << endl;
				HPofffreqfile << FreqCalcHPoff(genotype,rs) << endl;
			}
		}
	}
	return 0;
}
