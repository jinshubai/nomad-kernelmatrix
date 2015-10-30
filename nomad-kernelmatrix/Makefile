#CXX=g++
MPICXX=mpicxx

#DEBUGCXXFLAGS = -fPIC -O3 -fopenmp -std=c++11
CXXFLAGS = -fPIC -O3 -fopenmp -std=c++11 -Wall -fpermissive -g -gstabs+
#CXXFLAGS = -fPIC -O3 -DNDEBUG -fopenmp -std=c++11 

# TBB related paths
TBBROOT = tbb
TBBLIB = -L${TBBROOT}/lib -ltbb -ltbbmalloc

INCLUDEPATH = -I${TBBROOT}/include
LIBS = ${TBBLIB} -lrt -lm

all: nomad-q split

nomad-q: Scheduler.o Scheduler.h nomad_q.cpp
	${MPICXX} ${CXXFLAGS} ${INCLUDEPATH} -o nomad-q nomad_q.cpp Scheduler.o ${LIBS}

split: split.o
	${MPICXX} ${CXXFLAGS} -o split split.o 

Scheduler.o: Scheduler.cpp Scheduler.h
	${MPICXX} ${CXXFLAGS} ${INCLUDEPATH} -c -o Scheduler.o Scheduler.cpp ${LIBS}

split.o: split.cpp split.h
	${MPICXX} ${CXXFLAGS} -c -o split.o split.cpp

clean:
	rm -rf *.o nomad-q split