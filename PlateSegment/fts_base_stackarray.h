/*
 * fts_base_array.h
 *
 */
#ifndef _FTS_BASE_STACKARRAY_H_
#define _FTS_BASE_STACKARRAY_H_

// Usage:
//    int n = 10;
//   FTS_BASE_STACK_ARRAY( int, oIntArray, n );  // oIntArray is of type FTS_BASE_StackArray of size n
// NOTE: Can't use constant template parameter to specify the size of the array since
// the constant have to be resolved at compile time.

#ifndef WIN32
#define FTS_BASE_STACK_ARRAY( Type, Size, Name ) Type _##Name[ (Size) ]; FTS_BASE_StackArray<Type> Name( _##Name, (Size) )
#else
#define FTS_BASE_STACK_ARRAY( Type, Size, Name ) FTS_BASE_StackArray<Type> Name( (Size) )
#endif


/*!
 *
 * Wraps an array allocated on the stack so we can have boundary tests done on it.
 *
 * This class is meant to be lean and mean, so DON'T derive from it, otherwise
 * you can't guarantee performance.
 *
 * This class simulates the interface to std::vector
 *
 *
 */
template< class T >
class FTS_BASE_StackArray
{

public:

    // Constructors / Destructor / Assignment Operator / Copy Constructor
#ifndef WIN32
    explicit FTS_BASE_StackArray( T* poArray, const unsigned int nSize );
#else
	explicit FTS_BASE_StackArray( const unsigned int nSize );
#endif

    // Don't need a virtual destructor since this object cannot be allodated
    // on the heap.
    ~FTS_BASE_StackArray();


public:

    inline unsigned int size() const;
    inline T& at( unsigned int n );
    inline T* begin();
    inline T* end();
    inline T& front();
    inline T& back();


private:
#ifndef WIN32
    T* const m_poArray;
#else	
	T* m_poArray;
#endif	
    const unsigned int m_nSize;


private:

    // No copying
    FTS_BASE_StackArray( const FTS_BASE_StackArray& );
    FTS_BASE_StackArray& operator=( const FTS_BASE_StackArray& );

    // No heap allocation
    void* operator     new ( std::size_t );
    void* operator    new[]( std::size_t );
    void  operator delete  ( void* );
    void  operator delete[]( void* );

};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Class:  FTS_BASE_StackArray
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#ifndef WIN32
template< class T >
FTS_BASE_StackArray<T>::FTS_BASE_StackArray( T* poArray, const unsigned int nSize )
    : m_poArray( poArray )
    , m_nSize( nSize )
{
    assert( nSize != 0 );
}
#else
template< class T >
FTS_BASE_StackArray<T>::FTS_BASE_StackArray( const unsigned int nSize )    
    : m_poArray( NULL )
    , m_nSize( nSize )
{
    assert( nSize != 0 );
	m_poArray = new T[nSize];
}
#endif


template< class T >
FTS_BASE_StackArray<T>::~FTS_BASE_StackArray()
{
    // Nothing
#ifdef WIN32
	if(m_poArray)
		delete m_poArray;
#endif
}


template< class T >
inline T& FTS_BASE_StackArray<T>::at( unsigned int n )
{
    assert( n < m_nSize );

    return m_poArray[ n ];
}

template< class T >
inline unsigned int FTS_BASE_StackArray<T>::size() const
{
    return m_nSize;
}

template< class T >
inline T* FTS_BASE_StackArray<T>::begin()
{
    return m_poArray;
}

template< class T >
inline T* FTS_BASE_StackArray<T>::end()
{
    return m_poArray + m_nSize;
}

template< class T >
inline T& FTS_BASE_StackArray<T>::front()
{
    return m_poArray[ 0 ];
}

template< class T >
inline T& FTS_BASE_StackArray<T>::back()
{
    return m_poArray[ m_nSize-1 ];
}



#endif // _FTS_BASE_STACKARRAY_H_














