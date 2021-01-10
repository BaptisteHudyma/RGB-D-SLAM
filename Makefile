CC = g++

OPENCV = `pkg-config --cflags --libs opencv4`
EIGEN = `pkg-config --cflags --libs eigen3`

LIBS = $(OPENCV) $(EIGEN)
DEBUG = -g -O0
CFLAGS = -O4 -Wall

OUTFILE = ./build/

OBJ_CPP = $(wildcard *.cpp )
OBJ_O = $(patsubst %.cpp, $(OUTFILE)%.o, $(notdir ${OBJ_CPP}))

all: $(OBJ_O) 
	$(CC) $(CFLAGS) -o g.out $(OBJ_O) $(LIBS) 
	echo Finished

$(OUTFILE)%.o:%.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LIBS) 

clean:
	$(RM) $(OUTFILE)*
	echo "All Cleaned"

rmproper: clean
	$(RM) *.out
	echo "All removed"
