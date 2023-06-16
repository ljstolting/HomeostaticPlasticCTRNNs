main: main.o CTRNN.o random.o
	g++ -o main main.o CTRNN.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -c -O3 CTRNN.cpp
main.o: main.cpp CTRNN.h
	g++ -c -O3 main.cpp
clean:
	rm *.o main
