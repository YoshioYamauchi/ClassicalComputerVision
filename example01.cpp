// July 31, 2019
#define EXAMPLE01_CPP


#include <fstream>
#include <utility>
#include <vector>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "tbb/tbb.h"
using namespace tbb;


float Foo(float x)
{
  // printf("x=%f\n", x);

  return 3.14 * x;
}

// void ParallelApplyFoo(float* a, size_t n)
// {
//   // The third argument is a labmda function that takes a size_t type variable,
//   // and accesses all variables within the scope by reference.
//   tbb::parallel_for(size_t(0), n, [&](size_t i){
//     // printf("a[i] = %f\n", a[i]);
//     a[i] = Foo(a[i]);
//   });
//
// }

class ApplyFoo
{
  float* a_copy;
public:
  void operator()(const blocked_range<size_t>& r) const {
    float *a = a_copy;
    for(size_t i=r.begin(); i!=r.end(); ++i)
    {
      a[i] = Foo(a[i]);
    }
  }
  ApplyFoo(float* a):
    a_copy(a)
  {}
};

int main(int argc, char *argv[])
{
  // int grain_size = pow(2, 11);
  int grain_size = 416;
  // int size = pow(2, 20);
  int size = 416*416;
  float a[size];
  for(int i = 0; i<size; i++){a[i] = 1.0;}
  for(int i=0; i<10000; i++)
  {
    // one way to use parallel_for()
    // tbb::parallel_for(size_t(0), size_t(size), [&](size_t i){
    //   // printf("a[i] = %f\n", a[i]);
    //   a[i] = Foo(a[i]);
    // });

    // another way
    parallel_for(blocked_range<size_t>(0, size, grain_size), ApplyFoo(a));
  }


  // printf("a[size-1] = %f\n", a[size-1]);
  return 0;
}
