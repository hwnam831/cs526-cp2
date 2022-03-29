
#ifndef _STRM_H_
#define _STRM_H_

// Thread block size
#define BLOCK_SIZE 16

#define MW 1024
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA MW // Matrix A width
#define HA MW // Matrix A height
#define WB MW // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

#endif // _STRM_H_

