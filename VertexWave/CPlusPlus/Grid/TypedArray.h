// https://developer.mozilla.org/en/JavaScript_typed_arrays


template<typename T>
class TypedArray
{
public:
    TypedArray() :
        length(0),
        _buffer(0)
    {
    }

    TypedArray(unsigned long length) :
        length(length),
        _buffer(new T[length])
    {
    }

    ~TypedArray()
    {
        delete _buffer;
    }

    operator T*()
    {
        return _buffer;
    }

    operator const T*() const
    {
        return _buffer;
    }

    operator void*()
    {
        return _buffer;
    }

    operator const void*() const
    {
        return _buffer;
    }

    /* const */ unsigned long length;

    static const unsigned long BYTES_PER_ELEMENT = sizeof(T);

private:
    T *_buffer;
};

typedef TypedArray<float> Float32Array;
typedef TypedArray<double> Float64Array;
typedef TypedArray<short> Int16Array;
typedef TypedArray<int> Int32Array;
typedef TypedArray<char> Int8Array;
typedef TypedArray<unsigned short> Uint16Array;
typedef TypedArray<unsigned int> Uint32Array;
typedef TypedArray<unsigned char> Uint8Array;