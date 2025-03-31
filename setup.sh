source /opt/intel/oneapi/2025.1/oneapi-vars.sh
g++ -o main main.cpp -std=c++23 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -fopenmp -lboost_program_options -lm -ldl
./main --m1 1000 --n1 64 --m2 1000 --n2 64 --compute 1
