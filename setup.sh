source /opt/intel/oneapi/2025.1/oneapi-vars.sh
g++ -o main main.cpp -std=c++23 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -fopenmp -lboost_program_options -lm -ldl
./main --m1 3 --n1 2 --m2  3 --n2 2 --compute 1
# ./main --m1 1000 --n1 8 --m2  1000 --n2 8 --compute 1
