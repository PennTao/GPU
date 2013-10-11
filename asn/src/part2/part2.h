#ifndef PART2_H
#define PART2_h

#include "part2_kernel.h"

class ItemProcessor {
public:
    virtual void push(work_item_t item) = 0;
    virtual void clear() = 0;
    virtual void prepare(int num_to_expect, int len) = 0;
};

class BasicProcessor : public ItemProcessor {
public:
    virtual void push(work_item_t item);
    virtual void clear();
    virtual void prepare(int num_to_expect, int len);
private:
    basic_state_t state;
};

class AsyncProcessor : public ItemProcessor {
public:
    virtual void push(work_item_t item);
    virtual void clear();
    virtual void prepare(int num_to_expect, int len);
private:
    async_state_t state;
};

class StreamingProcessor : public ItemProcessor {
public:
    virtual void push(work_item_t item);
    virtual void clear();
    virtual void prepare(int num_to_expect, int len);
private:
    streaming_state_t state;
};

class MappedProcessor : public ItemProcessor {
public:
    virtual void push(work_item_t item);
    virtual void clear();
    virtual void prepare(int num_to_expect, int len);
private:
    mapped_state_t state;
};

#endif

