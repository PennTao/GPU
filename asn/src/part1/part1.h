#ifndef PART1_H
#define PART1_H

enum Processor {
    Host,
    Device
}; 

template <typename K>
void
qsortWrapper(K * i_data,
        K * o_data,
        int num,
        Processor src,
        Processor dst);

template <typename K>
void
stlWrapper(K * i_data,
        K * o_data,
        int num,
        Processor src,
        Processor dst);

template <typename K>
void
thrustWrapper(K * i_data, 
        K * o_data, 
        int num,
        Processor src,
        Processor dst);

#include "part1.inl"

#endif //PART1_H
